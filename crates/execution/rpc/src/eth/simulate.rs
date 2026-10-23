//! `eth_simulateV2`: `eth_simulateV1` under the block env the chain will actually execute.
//!
//! `eth_simulateV1` follows the execution-apis spec, which treats every simulated block as
//! hypothetical: it zeroes `prevRandao`, zeroes `basefee` whenever validation is off, and answers
//! `BLOCKHASH` for an earlier simulated block with that block's simulated hash. No block on Base
//! ever executes under those values, and contracts that read them branch on them rather than
//! degrade gracefully, e.g. hooks that seed a fee lottery from `PREVRANDAO` and
//! `BLOCKHASH(number - 1)` or price a tax off `BASEFEE`. A preview run under the placeholders takes
//! a branch the chain will not.
//!
//! `eth_simulateV2` takes the same request and returns the same response. It differs only where the
//! spec substitutes a placeholder:
//!
//! - `prevRandao` is the base block's, carried into every simulated block. Base inherits it from the
//!   L1 origin, so the next block repeats it unless the origin advances.
//! - `basefee` keeps the value derived from the parent, with or without validation. Validation off
//!   still disables the base fee check, so a call without a gas price still executes.
//! - `BLOCKHASH` of a block the request simulates fails the request with
//!   [`SimulatedBlockHashError`] instead of returning a hash no block on chain will have.
//!
//! Caller-supplied `blockOverrides` (`prevRandao`, `baseFeePerGas`, `blockHash`) still win.

use alloy_primitives::{Address, B256};
use alloy_rpc_types_eth::{
    BlockId,
    simulate::{SimulatePayload, SimulatedBlock},
};
use async_trait::async_trait;
use base_common_rpc_types::{BaseRpcTypes, BaseTransactionRequest};
use base_execution_chainspec::BaseChainSpec;
use base_execution_evm::BaseNextBlockEnvAttributes;
use jsonrpsee::proc_macros::rpc;
use jsonrpsee_core::RpcResult;
use reth_chainspec::ChainSpecProvider;
use reth_errors::ProviderError;
use reth_evm::ConfigureEvm;
use reth_rpc_eth_api::{FromEvmError, RpcBlock, RpcConvert, RpcNodeCore, helpers::SpawnBlocking};
use revm::{
    Database,
    primitives::{StorageKey, StorageValue},
    state::{AccountInfo, Bytecode},
};
use tracing::trace;

use crate::{BaseEthApi, BaseEthApiError};

/// Which simulate method a [`BaseEthApi`] simulation serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimulateVersion {
    /// `eth_simulateV1`, per the execution-apis spec.
    V1,
    /// `eth_simulateV2`, under the block env the chain will execute (see the module docs).
    V2,
}

/// `BLOCKHASH` requested for a block the simulation itself produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "BLOCKHASH({number}) requested for a simulated block, which has no hash on chain; pin one with \
     blockOverrides.blockHash"
)]
pub struct SimulatedBlockHashError {
    /// The block number whose hash was requested.
    pub number: u64,
}

/// Database that refuses `BLOCKHASH` from the first simulated block up.
///
/// Hashes pinned with `blockOverrides.blockHash` live in the enclosing `State`'s cache, which is
/// consulted before this database, so they still resolve.
#[derive(Debug)]
pub(crate) struct SealedBlockHashes<DB> {
    inner: DB,
    /// First block number without an on-chain hash; `None` passes every read through.
    first_simulated: Option<u64>,
}

impl<DB> SealedBlockHashes<DB> {
    /// Wraps `inner`, refusing block hashes from `first_simulated` up.
    pub(crate) const fn new(inner: DB, first_simulated: Option<u64>) -> Self {
        Self { inner, first_simulated }
    }
}

impl<DB: Database<Error = ProviderError>> Database for SealedBlockHashes<DB> {
    type Error = ProviderError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        self.inner.basic(address)
    }

    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        self.inner.code_by_hash(code_hash)
    }

    fn storage(
        &mut self,
        address: Address,
        index: StorageKey,
    ) -> Result<StorageValue, Self::Error> {
        self.inner.storage(address, index)
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        if self.first_simulated.is_some_and(|first| number >= first) {
            return Err(ProviderError::other(SimulatedBlockHashError { number }));
        }
        self.inner.block_hash(number)
    }
}

/// RPC endpoint for Base's `eth_simulateV2`.
#[cfg_attr(not(feature = "client"), rpc(server, namespace = "eth"))]
#[cfg_attr(feature = "client", rpc(server, client, namespace = "eth"))]
pub trait BaseEthSimulateApi {
    /// Executes calls on top of the requested state like `eth_simulateV1`, but under the block env
    /// the chain will execute: the base block's `prevRandao`, the derived `basefee` even with
    /// validation off, and an error instead of a simulated hash for `BLOCKHASH` of a simulated
    /// block.
    #[method(name = "simulateV2")]
    async fn simulate_v2(
        &self,
        opts: SimulatePayload<BaseTransactionRequest>,
        block_number: Option<BlockId>,
    ) -> RpcResult<Vec<SimulatedBlock<RpcBlock<BaseRpcTypes>>>>;
}

/// Handler for the `eth_simulateV2` RPC endpoint.
#[derive(Debug, Clone)]
pub struct BaseEthSimulateHandler<Eth> {
    eth_api: Eth,
}

impl<Eth> BaseEthSimulateHandler<Eth> {
    /// Creates a new [`BaseEthSimulateHandler`].
    pub const fn new(eth_api: Eth) -> Self {
        Self { eth_api }
    }
}

#[async_trait]
impl<N, Rpc> BaseEthSimulateApiServer for BaseEthSimulateHandler<BaseEthApi<N, Rpc>>
where
    N: RpcNodeCore,
    N::Evm: ConfigureEvm<NextBlockEnvCtx = BaseNextBlockEnvAttributes>,
    N::Provider: ChainSpecProvider<ChainSpec = BaseChainSpec>,
    BaseEthApiError: FromEvmError<N::Evm>,
    Rpc: RpcConvert<
            Primitives = N::Primitives,
            Error = BaseEthApiError,
            Evm = N::Evm,
            Network = BaseRpcTypes,
        >,
{
    async fn simulate_v2(
        &self,
        opts: SimulatePayload<BaseTransactionRequest>,
        block_number: Option<BlockId>,
    ) -> RpcResult<Vec<SimulatedBlock<RpcBlock<BaseRpcTypes>>>> {
        trace!(target: "rpc::eth", ?block_number, "Serving eth_simulateV2");
        // Same admission as reth's `eth_simulateV1` handler.
        let _permit = self.eth_api.tracing_task_guard().clone().acquire_owned().await;
        self.eth_api.simulate_v2(opts, block_number).await.map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use alloy_evm::overrides::OverrideBlockHashes;
    use revm::database::{EmptyDBTyped, State};

    use super::*;

    fn guarded(first_simulated: Option<u64>) -> SealedBlockHashes<EmptyDBTyped<ProviderError>> {
        SealedBlockHashes::new(EmptyDBTyped::new(), first_simulated)
    }

    #[test]
    fn refuses_hashes_from_the_first_simulated_block_up() {
        let mut db = guarded(Some(10));

        assert!(db.block_hash(9).is_ok());
        for number in [10, 11] {
            let err = db.block_hash(number).unwrap_err();
            assert!(err.to_string().contains(&format!("BLOCKHASH({number})")), "{err}");
        }
    }

    #[test]
    fn passes_every_hash_through_without_a_first_simulated_block() {
        let mut db = guarded(None);
        let mut inner = EmptyDBTyped::<ProviderError>::new();

        assert_eq!(db.block_hash(10).unwrap(), inner.block_hash(10).unwrap());
    }

    #[test]
    fn pinned_hashes_still_resolve() {
        let pin = B256::repeat_byte(0xcd);
        let mut state = State::builder().with_database(guarded(Some(10))).build();
        state.override_block_hashes(BTreeMap::from([(10, pin)]));

        assert_eq!(state.block_hash(10).unwrap(), pin);
        assert!(state.block_hash(11).is_err());
    }
}

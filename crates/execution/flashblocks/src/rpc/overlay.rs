//! Lazy pending-state overlay for flashblocks RPC.
//!
//! Pending `eth_fbSimulateV1` need to execute against the canonical state with the
//! flashblock diff applied on top. The historical approach converted the diff into a
//! [`StateOverride`](alloy_rpc_types_eth::state::StateOverride) and called
//! `apply_state_overrides`, which eagerly materializes *every* changed account and slot into the
//! database on every call — `O(block diff)` work (plus a canonical read per entry) regardless of
//! what the call actually touches.
//!
//! [`PendingBundleOverlay`] instead wraps the canonical state provider and serves reads lazily
//! from the already-in-memory [`BundleState`](revm::database::BundleState): each account/storage
//! lookup checks the bundle first (an `O(1)` hashmap hit) and falls through to canonical only on a
//! miss. Cost per call becomes `O(state the call reads)`, and there is no per-call clone of the
//! diff — the bundle is borrowed from the shared [`PendingBlocks`] snapshot.

use std::sync::Arc;

use alloy_consensus::BlockHeader;
use alloy_eips::BlockId;
use alloy_evm::overrides::{apply_block_overrides, apply_state_overrides};
use alloy_network::TransactionBuilder;
use alloy_primitives::{Address, B256, BlockNumber, StorageKey, StorageValue, U256};
use alloy_rpc_types::simulate::{SimBlock, SimulatePayload, SimulatedBlock};
use base_common_network::Base;
use base_common_rpc_types::BaseTransactionRequest;
use jsonrpsee::core::RpcResult;
use jsonrpsee_types::ErrorObjectOwned;
use reth_errors::RethError;
use reth_evm::{ConfigureEvm, Evm, env::BlockEnvironment, execute::BlockBuilder};
use reth_primitives_traits::{Account, Bytecode};
use reth_provider::ProviderResult;
use reth_revm::{
    database::{EvmStateProvider, StateProviderDatabase},
    db::{BundleState, State},
};
use reth_rpc_eth_api::{AsEthApiError, RpcBlock, helpers::FullEthApi};
use reth_rpc_eth_types::{
    EthApiError,
    error::FromEthApiError,
    simulate::{self, EthSimulateError},
};
use revm::{context::Block, primitives::KECCAK_EMPTY, state::AccountInfo};
use revm_inspectors::transfer::TransferInspector;

use crate::PendingBlocks;

/// A lazy overlay that serves reads from a flashblock [`BundleState`](revm::database::BundleState)
/// before falling back to the underlying canonical state provider `P`.
///
/// `P` is any [`EvmStateProvider`] — in production the boxed canonical state provider
/// (`StateProviderBox` satisfies [`EvmStateProvider`] via its blanket impl), and in tests a simple
/// in-memory mock.
#[derive(Debug, Clone)]
pub struct PendingBundleOverlay<P> {
    inner: P,
    bundle: Arc<BundleState>,
}

impl<P> PendingBundleOverlay<P> {
    /// Creates a new overlay over `inner` that serves reads from `bundle` before falling back to
    /// `inner`.
    pub const fn new(inner: P, bundle: Arc<BundleState>) -> Self {
        Self { inner, bundle }
    }
}

/// Converts a revm [`AccountInfo`] into a reth [`Account`].
fn to_reth_account(info: AccountInfo) -> Account {
    Account {
        nonce: info.nonce,
        balance: info.balance,
        // An empty-code account has no bytecode hash in reth's representation.
        bytecode_hash: (info.code_hash != KECCAK_EMPTY).then_some(info.code_hash),
    }
}

impl<P: EvmStateProvider> EvmStateProvider for PendingBundleOverlay<P> {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        if let Some(account) = self.bundle.account(address) {
            // `info` is `None` for an account that was destroyed / does not exist.
            return Ok(account.account_info().map(to_reth_account));
        }
        self.inner.basic_account(address)
    }

    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        // Block hashes are not affected by the pending diff.
        self.inner.block_hash(number)
    }

    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        if let Some(bytecode) = self.bundle.contracts.get(code_hash) {
            return Ok(Some(Bytecode(bytecode.clone())));
        }
        self.inner.bytecode_by_hash(code_hash)
    }

    fn storage(
        &self,
        account: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        if let Some(bundle_account) = self.bundle.account(&account) {
            // `storage_slot` returns the post-state value for written slots, zero when the
            // account's storage is fully known (created/destroyed this block), or `None` when the
            // slot is untouched on an otherwise-changed account — in which case we read canonical.
            // The bundle key space is `U256`, while the provider key is `B256`.
            if let Some(value) = bundle_account.storage_slot(U256::from_be_bytes(storage_key.0)) {
                return Ok(Some(value));
            }
        }
        self.inner.storage(account, storage_key)
    }
}

/// Flashblock-aware overlay implementations of `eth`-namespace calls.
///
/// These mirror the standard `eth_*` calls but execute against the in-memory flashblock
/// [`BundleState`](revm::database::BundleState) via [`PendingBundleOverlay`], so callers see the
/// pending block's effects without the canonical chain having advanced.
#[derive(Debug)]
pub struct OverlayCall;

impl OverlayCall {
    /// `eth_fbSimulateV1`: simulates `opts` on top of a pending flashblock bundle.
    ///
    /// This is the `eth_simulateV1` counterpart, but instead of converting the flashblock diff
    /// into a [`StateOverride`](alloy_rpc_types_eth::state::StateOverride) and materializing it on
    /// every call, it layers the pending [`BundleState`](revm::database::BundleState) lazily over
    /// the canonical state the bundle was built on using [`PendingBundleOverlay`], then runs the
    /// same per-block simulation loop as reth's `eth_simulateV1`.
    ///
    /// When both `block_number` and `block_index` are supplied, the bundle captured at that
    /// flashblock snapshot is used (returning [`EthApiError::HeaderNotFound`] if it is no longer
    /// retained); otherwise the latest bundle is used. Every snapshot of a given pending sequence
    /// is a diff over the same canonical parent, so the simulation base block is always
    /// [`canonical_block_number`](PendingBlocks::canonical_block_number) regardless.
    pub async fn simulate_v1<Eth>(
        eth_api: &Eth,
        pending: Arc<PendingBlocks>,
        opts: SimulatePayload<BaseTransactionRequest>,
        block_number: Option<u64>,
        block_index: Option<u64>,
    ) -> RpcResult<Vec<SimulatedBlock<RpcBlock<Eth::NetworkTypes>>>>
    where
        Eth: FullEthApi<NetworkTypes = Base> + Clone + Send + Sync + 'static,
        jsonrpsee_types::error::ErrorObject<'static>: From<Eth::Error>,
    {
        if opts.block_state_calls.len() > eth_api.max_simulate_blocks() as usize {
            return Err(EthApiError::InvalidParams("too many blocks.".to_string()).into());
        }
        if opts.block_state_calls.is_empty() {
            return Err(EthApiError::InvalidParams("calls are empty.".to_string()).into());
        }

        // Pick the bundle to overlay: a specific flashblock snapshot when both coordinates are
        // given, otherwise the latest.
        let bundle = match (block_number, block_index) {
            (Some(number), Some(index)) => {
                pending.get_historical_bundle_state_at(number, index).ok_or_else(|| {
                    let err: ErrorObjectOwned =
                        EthApiError::HeaderNotFound(BlockId::number(number)).into();
                    err
                })?
            }
            _ => pending.latest_bundle_state(),
        };

        // Simulate on top of the canonical block the flashblock bundle was built on; the bundle
        // diff is layered over that canonical state by `PendingBundleOverlay`.
        let block_id: BlockId = pending.canonical_block_number().into();

        let base_block =
            eth_api.recovered_block(block_id).await.map_err(Into::into)?.ok_or_else(|| {
                let err: ErrorObjectOwned = EthApiError::HeaderNotFound(block_id).into();
                err
            })?;
        let parent = base_block.sealed_header().clone();

        eth_api
            .spawn_blocking_io_fut(move |this| async move {
                let state = this.state_at_block_id(block_id).await?;
                let overlay = PendingBundleOverlay::new(state, bundle);
                let mut db =
                    State::builder().with_database(StateProviderDatabase::new(overlay)).build();

                let SimulatePayload {
                    block_state_calls,
                    trace_transfers,
                    validation,
                    return_full_transactions,
                } = opts;

                let mut parent = parent;
                let mut blocks: Vec<SimulatedBlock<RpcBlock<Eth::NetworkTypes>>> =
                    Vec::with_capacity(block_state_calls.len());

                // Track previous block number and timestamp for validation.
                let mut prev_block_number = parent.number();
                let mut prev_timestamp = parent.timestamp();

                for block in block_state_calls {
                    // Validate block number ordering if overridden.
                    if let Some(number) = block.block_overrides.as_ref().and_then(|o| o.number) {
                        let number: u64 = number.try_into().unwrap_or(u64::MAX);
                        if number <= prev_block_number {
                            return Err(Eth::Error::from_eth_err(EthApiError::other(
                                EthSimulateError::BlockNumberInvalid {
                                    got: number,
                                    parent: prev_block_number,
                                },
                            )));
                        }
                    }
                    // Validate timestamp ordering if overridden.
                    if let Some(time) = block
                        .block_overrides
                        .as_ref()
                        .and_then(|o| o.time)
                        .filter(|&t| t <= prev_timestamp)
                    {
                        return Err(Eth::Error::from_eth_err(EthApiError::other(
                            EthSimulateError::BlockTimestampInvalid {
                                got: time,
                                parent: prev_timestamp,
                            },
                        )));
                    }

                    let mut evm_env = this
                        .evm_config()
                        .next_evm_env(&parent, &this.next_env_attributes(&parent)?)
                        .map_err(RethError::other)
                        .map_err(Eth::Error::from_eth_err)?;

                    // Always disable EIP-3607.
                    evm_env.cfg_env.disable_eip3607 = true;

                    if !validation {
                        // If not explicitly required, disable nonce/fee checks.
                        evm_env.cfg_env.disable_nonce_check = true;
                        evm_env.cfg_env.disable_base_fee = true;
                        evm_env.cfg_env.tx_gas_limit_cap = Some(u64::MAX);
                        evm_env.block_env.inner_mut().basefee = 0;
                    }

                    let SimBlock { block_overrides, state_overrides, calls } = block;

                    // Set prevrandao to zero for simulated blocks by default; a user override is
                    // applied by `apply_block_overrides`.
                    evm_env.block_env.inner_mut().prevrandao = Some(B256::ZERO);

                    if let Some(block_overrides) = block_overrides {
                        // Ensure we don't allow an uncapped gas limit per block.
                        if let Some(gas_limit_override) = block_overrides.gas_limit
                            && gas_limit_override > evm_env.block_env.gas_limit()
                            && gas_limit_override > this.call_gas_limit()
                        {
                            return Err(Eth::Error::from_eth_err(EthApiError::other(
                                EthSimulateError::GasLimitReached,
                            )));
                        }
                        apply_block_overrides(
                            block_overrides,
                            &mut db,
                            evm_env.block_env.inner_mut(),
                        );
                    }
                    if let Some(ref state_overrides) = state_overrides {
                        apply_state_overrides(state_overrides.clone(), &mut db)
                            .map_err(Eth::Error::from_eth_err)?;
                    }

                    let block_gas_limit = evm_env.block_env.gas_limit();
                    let chain_id = evm_env.cfg_env.chain_id;

                    let default_gas_limit = {
                        let total_specified_gas =
                            calls.iter().filter_map(|tx| tx.as_ref().gas_limit()).sum::<u64>();
                        let txs_without_gas_limit =
                            calls.iter().filter(|tx| tx.as_ref().gas_limit().is_none()).count();

                        if total_specified_gas > block_gas_limit {
                            return Err(Eth::Error::from_eth_err(EthApiError::Other(Box::new(
                                EthSimulateError::BlockGasLimitExceeded,
                            ))));
                        }

                        if txs_without_gas_limit > 0 {
                            // Divide remaining gas equally among transactions without gas.
                            let gas_per_tx = (block_gas_limit - total_specified_gas)
                                / txs_without_gas_limit as u64;
                            let call_gas_limit = this.call_gas_limit();
                            if call_gas_limit > 0 {
                                gas_per_tx.min(call_gas_limit)
                            } else {
                                gas_per_tx
                            }
                        } else {
                            0
                        }
                    };

                    let ctx = this
                        .evm_config()
                        .context_for_next_block(&parent, this.next_env_attributes(&parent)?)
                        .map_err(RethError::other)
                        .map_err(Eth::Error::from_eth_err)?;
                    let map_err = |e: EthApiError| -> Eth::Error {
                        e.as_simulate_error().map_or_else(
                            || Eth::Error::from_eth_err(e),
                            |sim_err| Eth::Error::from_eth_err(EthApiError::other(sim_err)),
                        )
                    };

                    let (result, results) = if trace_transfers {
                        // Capture transfers inside the EVM so they are recorded as logs.
                        let inspector = TransferInspector::new(false).with_logs(true);
                        let evm = this
                            .evm_config()
                            .evm_with_env_and_inspector(&mut db, evm_env, inspector);
                        let mut builder = this.evm_config().create_block_builder(evm, &parent, ctx);

                        if let Some(ref state_overrides) = state_overrides {
                            simulate::apply_precompile_overrides(
                                state_overrides,
                                builder.evm_mut().precompiles_mut(),
                            )
                            .map_err(|e| Eth::Error::from_eth_err(EthApiError::other(e)))?;
                        }

                        simulate::execute_transactions(
                            builder,
                            calls,
                            default_gas_limit,
                            chain_id,
                            this.converter(),
                        )
                        .map_err(map_err)?
                    } else {
                        let evm = this.evm_config().evm_with_env(&mut db, evm_env);
                        let mut builder = this.evm_config().create_block_builder(evm, &parent, ctx);

                        if let Some(ref state_overrides) = state_overrides {
                            simulate::apply_precompile_overrides(
                                state_overrides,
                                builder.evm_mut().precompiles_mut(),
                            )
                            .map_err(|e| Eth::Error::from_eth_err(EthApiError::other(e)))?;
                        }

                        simulate::execute_transactions(
                            builder,
                            calls,
                            default_gas_limit,
                            chain_id,
                            this.converter(),
                        )
                        .map_err(map_err)?
                    };

                    parent = result.block.clone_sealed_header();

                    // Update tracking for the next iteration's validation.
                    prev_block_number = parent.number();
                    prev_timestamp = parent.timestamp();

                    let block = simulate::build_simulated_block::<Eth::Error, _>(
                        result.block,
                        results,
                        return_full_transactions.into(),
                        this.converter(),
                    )?;

                    blocks.push(block);
                }

                Ok(blocks)
            })
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use alloy_consensus::{Header, Sealed};
    use alloy_rpc_types_engine::PayloadId;
    use base_common_flashblocks::{
        ExecutionPayloadBaseV1, ExecutionPayloadFlashblockDeltaV1, Flashblock, Metadata,
    };
    use revm::{
        database::{AccountStatus, BundleAccount, BundleState, states::StorageSlot},
        state::AccountInfo,
    };

    use super::*;
    use crate::PendingBlocksBuilder;

    /// A trivial in-memory canonical state used as the overlay fallback in tests.
    #[derive(Debug, Default)]
    struct MockState {
        accounts: HashMap<Address, Account>,
        storage: HashMap<(Address, StorageKey), StorageValue>,
    }

    impl EvmStateProvider for MockState {
        fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
            Ok(self.accounts.get(address).copied())
        }
        fn block_hash(&self, _number: BlockNumber) -> ProviderResult<Option<B256>> {
            Ok(None)
        }
        fn bytecode_by_hash(&self, _code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
            Ok(None)
        }
        fn storage(
            &self,
            account: Address,
            storage_key: StorageKey,
        ) -> ProviderResult<Option<StorageValue>> {
            Ok(self.storage.get(&(account, storage_key)).copied())
        }
    }

    fn addr(b: u8) -> Address {
        Address::repeat_byte(b)
    }

    /// Minimal flashblock so `PendingBlocksBuilder::build` succeeds.
    fn minimal_flashblock() -> Flashblock {
        Flashblock {
            payload_id: PayloadId::default(),
            index: 0,
            base: Some(ExecutionPayloadBaseV1 {
                parent_beacon_block_root: B256::ZERO,
                parent_hash: B256::ZERO,
                fee_recipient: Address::ZERO,
                prev_randao: B256::ZERO,
                block_number: 1,
                gas_limit: 30_000_000,
                timestamp: 0,
                extra_data: Default::default(),
                base_fee_per_gas: U256::ZERO,
            }),
            diff: ExecutionPayloadFlashblockDeltaV1 {
                state_root: B256::ZERO,
                receipts_root: B256::ZERO,
                logs_bloom: Default::default(),
                gas_used: 0,
                block_hash: B256::ZERO,
                transactions: vec![],
                withdrawals: vec![],
                withdrawals_root: B256::ZERO,
                blob_gas_used: None,
            },
            metadata: Metadata { block_number: 1 },
        }
    }

    /// Builds a `PendingBlocks` whose bundle changes `account` (balance + one slot) with `status`.
    fn pending_with_change(
        account: Address,
        balance: u64,
        slot: B256,
        value: u64,
        status: AccountStatus,
    ) -> Arc<PendingBlocks> {
        let info = AccountInfo { balance: U256::from(balance), nonce: 1, ..Default::default() };
        let mut storage = HashMap::default();
        storage.insert(
            U256::from_be_bytes(slot.0),
            StorageSlot::new_changed(U256::ZERO, U256::from(value)),
        );

        let mut bundle = BundleState::default();
        bundle.state.insert(account, BundleAccount::new(None, Some(info), storage, status));

        let mut builder = PendingBlocksBuilder::new();
        builder.with_header(Sealed::new_unchecked(
            Header { number: 1, ..Default::default() },
            B256::ZERO,
        ));
        builder.with_flashblocks([minimal_flashblock()]);
        builder.with_bundle_state(bundle);
        Arc::new(builder.build(None).expect("build pending blocks"))
    }

    #[test]
    fn bundle_account_overrides_canonical() {
        let account = addr(1);
        let slot = B256::from(U256::from(7));
        let pending = pending_with_change(account, 500, slot, 42, AccountStatus::Changed);

        let mut canonical = MockState::default();
        canonical
            .accounts
            .insert(account, Account { nonce: 0, balance: U256::from(1), bytecode_hash: None });
        canonical.storage.insert((account, slot), U256::from(9));

        let overlay = PendingBundleOverlay::new(canonical, pending.latest_bundle_state());

        // Balance + slot come from the bundle, not canonical.
        assert_eq!(overlay.basic_account(&account).unwrap().unwrap().balance, U256::from(500));
        assert_eq!(overlay.storage(account, slot).unwrap(), Some(U256::from(42)));
    }

    #[test]
    fn falls_through_to_canonical_for_untouched_state() {
        let changed = addr(1);
        let slot = B256::from(U256::from(7));
        // `Changed` (not storage-known) => untouched slots fall through to canonical.
        let pending = pending_with_change(changed, 500, slot, 42, AccountStatus::Changed);

        let other = addr(2);
        let other_slot = B256::from(U256::from(3));
        let untouched_slot = B256::from(U256::from(99));
        let mut canonical = MockState::default();
        canonical
            .accounts
            .insert(other, Account { nonce: 5, balance: U256::from(123), bytecode_hash: None });
        canonical.storage.insert((other, other_slot), U256::from(77));
        canonical.storage.insert((changed, untouched_slot), U256::from(55));

        let overlay = PendingBundleOverlay::new(canonical, pending.latest_bundle_state());

        // Account not in the bundle: canonical.
        assert_eq!(overlay.basic_account(&other).unwrap().unwrap().balance, U256::from(123));
        assert_eq!(overlay.storage(other, other_slot).unwrap(), Some(U256::from(77)));
        // Untouched slot on a changed account: canonical.
        assert_eq!(overlay.storage(changed, untouched_slot).unwrap(), Some(U256::from(55)));
    }

    #[test]
    fn historical_bundle_snapshot_is_retained_and_keyed() {
        let account = addr(1);
        let slot = B256::from(U256::from(7));
        // `pending_with_change` builds a single flashblock at block 1, index 0.
        let pending = pending_with_change(account, 500, slot, 42, AccountStatus::Changed);

        // The snapshot at the latest (block, index) is retained and matches the latest bundle.
        let snapshot = pending.get_historical_bundle_state_at(1, 0).expect("snapshot retained");
        assert!(snapshot.account(&account).is_some());
        assert!(Arc::ptr_eq(&snapshot, &pending.latest_bundle_state()));

        // Coordinates with no snapshot return `None`.
        assert!(pending.get_historical_bundle_state_at(1, 9).is_none());
        assert!(pending.get_historical_bundle_state_at(2, 0).is_none());
    }
}

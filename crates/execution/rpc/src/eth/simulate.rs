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
//! `eth_simulateV2` takes the same request and returns the same response, each call with one more
//! field, `gasExtra` ([`BaseSimCallResult`]). It differs where the spec substitutes a placeholder,
//! and in that field:
//!
//! - `prevRandao` is the base block's, carried into every simulated block. Base inherits it from the
//!   L1 origin, so the next block repeats it unless the origin advances.
//! - `basefee` keeps the value derived from the parent, with or without validation. Validation off
//!   still disables the base fee check, so a call without a gas price still executes.
//! - `BLOCKHASH` of a block the request simulates fails the request with
//!   [`SimulatedBlockHashError`] instead of returning a hash no block on chain will have.
//! - A call that succeeds past a `gasleft()` check guarding a revert reports the check's bound, the
//!   largest one's for several, as its `gasExtra` (see [`GasLeftGuardInspector`]); `gasUsed` and
//!   `maxUsedGas` are what the call used. Such a check passes under the generous gas a preview runs
//!   with and can fail on chain under the limit a transaction is sent with, which the request does
//!   not know. `gasUsed + gasExtra` covers a check in the call's own frame; one in a frame `n`
//!   calls down needs up to about `n/64` of the bound more, since a call forwards at most 63/64 of
//!   the gas left (EIP-150).
//!
//! Caller-supplied `blockOverrides` (`prevRandao`, `baseFeePerGas`, `blockHash`) still win.

use std::{cell::RefCell, rc::Rc};

use alloy_eips::eip4788::SYSTEM_ADDRESS;
use alloy_primitives::{Address, B256, U256};
use alloy_rpc_types_eth::{
    BlockId,
    simulate::{SimCallResult, SimulatePayload, SimulatedBlock},
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
    Database, Inspector,
    bytecode::opcode,
    context_interface::{ContextTr, Transaction},
    interpreter::{
        CallInputs, CallOutcome, CreateInputs, CreateOutcome, Interpreter, InterpreterTypes,
        interpreter_types::{Jumps, LegacyBytecode, StackTr},
    },
    primitives::{StorageKey, StorageValue},
    state::{AccountInfo, Bytecode},
};
use serde::{Deserialize, Serialize};
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

/// The largest `bound` of each simulated transaction's `gasleft()` checks that guard a revert, in
/// execution order: `None` for a transaction without one. Filled by a [`GasLeftGuardInspector`]
/// holding a clone, and read once the block has run.
#[derive(Debug, Clone, Default)]
pub struct GasLeftBounds(pub Rc<RefCell<Vec<Option<u64>>>>);

impl GasLeftBounds {
    /// `calls` with their `gasExtra`: its bound for each call that succeeded, `None` for the rest.
    /// `None` for all of them unless there is one entry per call, as there is not for a block run
    /// without the inspector.
    pub fn attach(&self, calls: Vec<SimCallResult>) -> Vec<BaseSimCallResult> {
        let bounds = self.0.borrow();
        let matched = bounds.len() == calls.len();
        calls
            .into_iter()
            .enumerate()
            .map(|(index, inner)| {
                let gas_extra = if matched && inner.status { bounds[index] } else { None };
                BaseSimCallResult { inner, gas_extra }
            })
            .collect()
    }
}

/// A call of an `eth_simulateV2` block: what `eth_simulateV1` reports for it, and `gasExtra`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseSimCallResult {
    /// What `eth_simulateV1` reports for the call.
    #[serde(flatten)]
    pub inner: SimCallResult,
    /// The bound of the `gasleft()` checks guarding a revert that the call passed, the largest for
    /// several ([`GasLeftGuardInspector`]): about how much more gas than it uses a transaction
    /// sending the call needs. Absent for a call that failed or passed no such check.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "alloy_serde::quantity::opt")]
    pub gas_extra: Option<u64>,
}

/// A block of `eth_simulateV2`: the block `eth_simulateV1` reports, and its calls'
/// [`BaseSimCallResult`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseSimulatedBlock<B> {
    /// The simulated block.
    #[serde(flatten)]
    pub inner: B,
    /// The block's calls.
    pub calls: Vec<BaseSimCallResult>,
}

impl<B> BaseSimulatedBlock<B> {
    /// The block as `eth_simulateV1` reports it, without `gasExtra`.
    pub fn into_v1(self) -> SimulatedBlock<B> {
        SimulatedBlock {
            inner: self.inner,
            calls: self.calls.into_iter().map(|call| call.inner).collect(),
        }
    }
}

/// Finds the `gasleft()` checks that guard a revert. Execution is left as it is.
///
/// A `GAS` read is kept until the frame calls out or ends. A comparison (`LT`, `GT`, `SLT`, `SGT`)
/// of that value with anything is a [`GasCheck`] when its result decides the next `JUMPI`, through
/// any `ISZERO`s and the push of the jump target, and the branch it rules out reaches `REVERT` or
/// `INVALID` ([`Self::reverts_from`]): `require(gasleft() >= bound)` and
/// `if (gasleft() < bound) revert ...` compile to one, with `bound` a constant or a stored value.
/// When a transaction ends, the largest bound of its checks goes to the [`GasLeftBounds`].
#[derive(Debug, Default, Clone)]
pub struct GasLeftGuardInspector {
    /// Where each transaction's bound goes.
    bounds: GasLeftBounds,
    /// The previous step was a `GAS`, whose value is now on top of the stack.
    gas_pushed: bool,
    /// What the current frame's last `GAS` read.
    gas_left: Option<U256>,
    /// A comparison of `gas_left` whose result has not reached a `JUMPI` yet.
    check: Option<GasCheck>,
    /// Frames entered and not yet left, the transaction's own frame first.
    depth: usize,
    /// The running transaction is one of the block's, not a system call, which has no entry.
    counted: bool,
    /// The largest bound of the running transaction's checks so far.
    bound: Option<u64>,
}

/// A comparison of `gasleft()` with `bound`, whose result is on the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasCheck {
    /// The `gasleft()` compared.
    pub gas_left: U256,
    /// What it was compared with.
    pub bound: U256,
    /// Stack index of the comparison's result.
    pub result: usize,
}

impl GasLeftGuardInspector {
    /// Instructions [`Self::reverts_from`] reads before it gives up. Solidity's revert paths are
    /// short: 9 instructions for `InsufficientGas()` in hook
    /// `0x51ab3586aa9caea232d2e34fa34d10b777232840`, 28 for the `Error(string)` in implementation
    /// `0x40365a9b0f3fa00cd34e28d9582799501255b371`.
    pub const MAX_WALK: usize = 64;

    /// An inspector that writes each transaction's bound to `bounds`.
    pub fn new(bounds: GasLeftBounds) -> Self {
        Self { bounds, ..Self::default() }
    }

    /// Enters a frame: the transaction's own when none is open.
    pub fn enter<CTX: ContextTr>(&mut self, context: &CTX) {
        if self.depth == 0 {
            self.counted = context.tx().caller() != SYSTEM_ADDRESS;
            self.bound = None;
        }
        self.depth += 1;
        self.forget_reads();
    }

    /// Leaves a frame: the transaction's own when it was the last open, which records its bound.
    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
        self.forget_reads();
        if self.depth == 0 && std::mem::take(&mut self.counted) {
            self.bounds.0.borrow_mut().push(self.bound.take());
        }
    }

    /// Forgets the `GAS` read and comparison of the frame being left or suspended.
    pub const fn forget_reads(&mut self) {
        self.gas_pushed = false;
        self.gas_left = None;
        self.check = None;
    }

    /// Whether execution from `pc` reaches `REVERT` or `INVALID` within [`Self::MAX_WALK`]
    /// instructions. It follows a `JUMP` to the target pushed right before it; any other `JUMP`, a
    /// `JUMPI`, or an exit without a revert ends the walk.
    pub fn reverts_from<B: Jumps + LegacyBytecode>(bytecode: &mut B, mut pc: usize) -> bool {
        let mut pushed = None;
        for _ in 0..Self::MAX_WALK {
            let Some(&op) = bytecode.bytecode_slice().get(pc) else { return false };
            match op {
                opcode::REVERT | opcode::INVALID => return true,
                opcode::STOP | opcode::RETURN | opcode::SELFDESTRUCT | opcode::JUMPI => {
                    return false;
                }
                opcode::JUMP => match pushed {
                    Some(target) if bytecode.is_valid_legacy_jump(target) => {
                        pc = target;
                        pushed = None;
                    }
                    _ => return false,
                },
                opcode::PUSH0..=opcode::PUSH32 => {
                    let size = usize::from(op - opcode::PUSH0);
                    pushed = bytecode
                        .bytecode_slice()
                        .get(pc + 1..pc + 1 + size)
                        .and_then(|immediate| usize::try_from(U256::from_be_slice(immediate)).ok());
                    pc += 1 + size;
                }
                _ => {
                    pushed = None;
                    pc += 1;
                }
            }
        }
        false
    }
}

impl<CTX: ContextTr, INTR: InterpreterTypes> Inspector<CTX, INTR> for GasLeftGuardInspector {
    fn step(&mut self, interp: &mut Interpreter<INTR>, _context: &mut CTX) {
        if self.gas_pushed {
            self.gas_pushed = false;
            self.gas_left = interp.stack.data().last().copied();
        }
        let op = interp.bytecode.opcode();
        if let Some(check) = self.check.take() {
            let depth = interp.stack.len();
            match op {
                opcode::ISZERO if depth == check.result + 1 => self.check = Some(check),
                opcode::PUSH0..=opcode::PUSH32 => self.check = Some(check),
                // A check `gasleft()` passes from below guards nothing a lower limit breaks.
                opcode::JUMPI if depth == check.result + 2 && check.gas_left >= check.bound => {
                    let [.., condition, target] = interp.stack.data() else { return };
                    // The branch the check rules out: the fall-through when it jumps, the target
                    // when it does not.
                    let ruled_out = if condition.is_zero() {
                        usize::try_from(*target)
                            .ok()
                            .filter(|&target| interp.bytecode.is_valid_legacy_jump(target))
                    } else {
                        Some(interp.bytecode.pc() + 1)
                    };
                    if ruled_out.is_some_and(|pc| Self::reverts_from(&mut interp.bytecode, pc)) {
                        let bound = u64::try_from(check.bound).unwrap_or(u64::MAX);
                        self.bound = Some(self.bound.map_or(bound, |largest| largest.max(bound)));
                    }
                    return;
                }
                _ => {}
            }
        }
        match op {
            opcode::GAS => self.gas_pushed = true,
            opcode::LT | opcode::GT | opcode::SLT | opcode::SGT => {
                let Some(gas_left) = self.gas_left else { return };
                let [.., second, top] = interp.stack.data() else { return };
                let bound = match (*top == gas_left, *second == gas_left) {
                    (true, false) => *second,
                    (false, true) => *top,
                    _ => return,
                };
                let result = interp.stack.len() - 2;
                self.check = Some(GasCheck { gas_left, bound, result });
            }
            _ => {}
        }
    }

    fn call(&mut self, context: &mut CTX, _inputs: &mut CallInputs) -> Option<CallOutcome> {
        self.enter(context);
        None
    }

    fn call_end(&mut self, _context: &mut CTX, _inputs: &CallInputs, _outcome: &mut CallOutcome) {
        self.leave();
    }

    fn create(&mut self, context: &mut CTX, _inputs: &mut CreateInputs) -> Option<CreateOutcome> {
        self.enter(context);
        None
    }

    fn create_end(
        &mut self,
        _context: &mut CTX,
        _inputs: &CreateInputs,
        _outcome: &mut CreateOutcome,
    ) {
        self.leave();
    }
}

/// RPC endpoint for Base's `eth_simulateV2`.
#[cfg_attr(not(feature = "client"), rpc(server, namespace = "eth"))]
#[cfg_attr(feature = "client", rpc(server, client, namespace = "eth"))]
pub trait BaseEthSimulateApi {
    /// Executes calls on top of the requested state like `eth_simulateV1`, but under the block env
    /// the chain will execute: the base block's `prevRandao`, the derived `basefee` even with
    /// validation off, and an error instead of a simulated hash for `BLOCKHASH` of a simulated
    /// block. A call that succeeds past a `gasleft()` check guarding a revert reports the check's
    /// bound as its `gasExtra`.
    #[method(name = "simulateV2")]
    async fn simulate_v2(
        &self,
        opts: SimulatePayload<BaseTransactionRequest>,
        block_number: Option<BlockId>,
    ) -> RpcResult<Vec<BaseSimulatedBlock<RpcBlock<BaseRpcTypes>>>>;
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
    ) -> RpcResult<Vec<BaseSimulatedBlock<RpcBlock<BaseRpcTypes>>>> {
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
    use alloy_primitives::{Bytes, TxKind, address, hex};
    use revm::{
        Context, ExecuteEvm, InspectEvm, MainBuilder, MainContext,
        context::{TxEnv, result::ExecutionResult},
        database::{CacheDB, EmptyDB, EmptyDBTyped, State},
    };

    use super::*;

    const CALLER: Address = address!("0x0000000000000000000000000000000000000ca1");
    const HOOK: Address = address!("0x000000000000000000000000000000000000f100");
    const POOL_MANAGER: Address = address!("0x000000000000000000000000000000000000f200");
    const GAS_LIMIT: u64 = 1_000_000;

    /// Slot [`STORED_FLOOR`] reads its bound from.
    const MIN_GAS_LEFT_SLOT: u64 = 5;

    /// `if (gasleft() < 500000) revert InsufficientGas();`, in the instructions of hook
    /// `0x51ab3586aa9caea232d2e34fa34d10b777232840`: `PUSH3 500000 GAS LT PUSH2 JUMPI` to the
    /// revert.
    const FLOOR: [u8; 26] = hex!("6207a1205a1061000b57005b6307099c5360e21b5f5260045ffd");
    /// `require(gasleft() >= minGasLeft)`, in the instructions of implementation
    /// `0x40365a9b0f3fa00cd34e28d9582799501255b371`: `GAS PUSH1 5 SLOAD GT ISZERO PUSH2 JUMPI`
    /// past the revert it falls through to.
    const STORED_FLOOR: [u8; 16] = hex!("5a600554111561000e57600080fd5b00");
    /// [`FLOOR`] with the revert behind a helper the failing branch jumps to.
    const HELPER_REVERT: [u8; 23] = hex!("6207a1205a1061000b57005b610000610013565b5f5ffd");
    /// `gasleft() < 500000` choosing between two exits, neither a revert.
    const NO_REVERT: [u8; 13] = hex!("6207a1205a1061000b57005b00");
    /// `1000000 < 500000` guarding a revert: a check that does not read `gasleft()`.
    const NO_GAS: [u8; 18] = hex!("6207a120620f42401061000e57005b5f5ffd");
    /// `gasleft() < 500000` returned rather than branched on.
    const RETURNED: [u8; 14] = hex!("6207a1205a1060005260206000f3");

    /// Calls [`HOOK`] with all its gas and reverts with whatever the hook reverted with, as the
    /// pool manager does around a hook callback.
    fn pool_manager() -> Vec<u8> {
        [
            &hex!("6000600060006000600073")[..],
            HOOK.as_slice(),
            &hex!("5af1602e573d600060003e3d6000fd5b00"),
        ]
        .concat()
    }

    /// `contracts` deployed, and [`HOOK`]'s [`MIN_GAS_LEFT_SLOT`] holding `min_gas_left`.
    fn db(contracts: &[(Address, &[u8])], min_gas_left: u64) -> CacheDB<EmptyDB> {
        let mut db = CacheDB::<EmptyDB>::default();
        for (address, code) in contracts {
            let code = Bytecode::new_raw(Bytes::copy_from_slice(code));
            db.insert_account_info(*address, AccountInfo::default().with_code(code));
        }
        db.insert_account_storage(HOOK, U256::from(MIN_GAS_LEFT_SLOT), U256::from(min_gas_left))
            .unwrap();
        db
    }

    /// A call to `target` carrying `gas_limit`.
    fn call(target: Address, gas_limit: u64) -> TxEnv {
        TxEnv::builder()
            .caller(CALLER)
            .kind(TxKind::Call(target))
            .gas_limit(gas_limit)
            .build()
            .unwrap()
    }

    /// Runs a call to `target` carrying [`GAS_LIMIT`] under the inspector: its result and bound.
    fn inspect(
        contracts: &[(Address, &[u8])],
        min_gas_left: u64,
        target: Address,
    ) -> (ExecutionResult, Option<u64>) {
        let bounds = GasLeftBounds::default();
        let mut evm = Context::mainnet()
            .with_db(db(contracts, min_gas_left))
            .build_mainnet_with_inspector(GasLeftGuardInspector::new(bounds.clone()));
        let result = evm.inspect_one_tx(call(target, GAS_LIMIT)).unwrap();
        let bounds = bounds.0.take();
        assert_eq!(bounds.len(), 1, "one entry per transaction");
        (result, bounds[0])
    }

    /// The least gas limit with which a call to `target` succeeds, run uninspected as the chain
    /// runs it.
    fn least_passing(contracts: &[(Address, &[u8])], min_gas_left: u64, target: Address) -> u64 {
        let succeeds = |gas_limit| {
            let mut evm = Context::mainnet().with_db(db(contracts, min_gas_left)).build_mainnet();
            evm.transact(call(target, gas_limit)).unwrap().result.is_success()
        };
        // Bare intrinsic gas runs no code, so the call fails there.
        let (mut fails, mut passes) = (21_000, GAS_LIMIT);
        assert!(succeeds(passes));
        while passes - fails > 1 {
            let mid = fails.midpoint(passes);
            if succeeds(mid) { passes = mid } else { fails = mid }
        }
        passes
    }

    #[test]
    fn gas_used_plus_the_bound_passes_a_check_in_the_call_itself() {
        for (code, min_gas_left) in [(&FLOOR[..], 0), (&STORED_FLOOR, 600_000), (&HELPER_REVERT, 0)]
        {
            let contracts = [(HOOK, code)];
            let (result, bound) = inspect(&contracts, min_gas_left, HOOK);
            assert!(result.is_success(), "{result:?}");
            assert_eq!(bound, Some(min_gas_left.max(500_000)));
            let limit = result.tx_gas_used() + bound.unwrap();
            assert!(least_passing(&contracts, min_gas_left, HOOK) <= limit);
        }
    }

    #[test]
    fn twice_gas_used_plus_the_bound_passes_a_check_a_call_below() {
        let pool_manager = pool_manager();
        let contracts = [(POOL_MANAGER, &pool_manager[..]), (HOOK, &FLOOR)];
        let (result, bound) = inspect(&contracts, 0, POOL_MANAGER);
        assert!(result.is_success(), "{result:?}");
        assert_eq!(bound, Some(500_000));
        // A call forwards 63/64 of the gas left, so the sum alone falls short a level down; the
        // engine's twice the estimate covers it.
        let limit = 2 * (result.tx_gas_used() + 500_000);
        assert!(least_passing(&contracts, 0, POOL_MANAGER) <= limit);
    }

    #[test]
    fn no_bound_for_a_check_that_guards_no_revert() {
        for code in [&NO_REVERT[..], &NO_GAS, &RETURNED] {
            let (result, bound) = inspect(&[(HOOK, code)], 0, HOOK);
            assert!(result.is_success(), "{result:?}");
            assert_eq!(bound, None);
        }
        let pool_manager = pool_manager();
        let (result, bound) =
            inspect(&[(POOL_MANAGER, &pool_manager), (HOOK, &NO_REVERT)], 0, POOL_MANAGER);
        assert!(result.is_success(), "{result:?}");
        assert_eq!(bound, None);
    }

    #[test]
    fn no_bound_for_a_check_that_already_fails() {
        let (result, bound) = inspect(&[(HOOK, &STORED_FLOOR)], 1_500_000, HOOK);
        assert!(!result.is_success(), "{result:?}");
        assert_eq!(bound, None);
    }

    #[test]
    fn bounds_line_up_with_transactions() {
        let other = address!("0x000000000000000000000000000000000000f300");
        let bounds = GasLeftBounds::default();
        let mut evm = Context::mainnet()
            .with_db(db(&[(HOOK, &FLOOR), (other, &NO_REVERT)], 0))
            .build_mainnet_with_inspector(GasLeftGuardInspector::new(bounds.clone()));
        for (nonce, target) in (0..).zip([HOOK, other, HOOK]) {
            evm.inspect_one_tx(TxEnv { nonce, ..call(target, GAS_LIMIT) }).unwrap();
        }
        assert_eq!(bounds.0.take(), [Some(500_000), None, Some(500_000)]);
    }

    /// A call that ran, burning 50k gas, or that reverted.
    fn sim_call(status: bool) -> SimCallResult {
        SimCallResult {
            return_data: Bytes::new(),
            logs: Vec::new(),
            gas_used: 50_000,
            max_used_gas: Some(60_000),
            status,
            error: None,
        }
    }

    #[test]
    fn gas_extra_is_a_succeeded_calls_bound() {
        let calls = || vec![sim_call(true), sim_call(true), sim_call(false)];
        let bounds = GasLeftBounds::default();
        bounds.0.borrow_mut().extend([Some(500_000), None, Some(500_000)]);
        let attached = bounds.attach(calls());
        let gas_extra: Vec<_> = attached.iter().map(|call| call.gas_extra).collect();
        assert_eq!(gas_extra, [Some(500_000), None, None]);
        let gas = |call: &SimCallResult| (call.status, call.gas_used, call.max_used_gas);
        let reported: Vec<_> = attached.iter().map(|call| gas(&call.inner)).collect();
        let used: Vec<_> = calls().iter().map(gas).collect();
        assert_eq!(reported, used, "gasUsed and maxUsedGas are left as they are");

        // A block run without the inspector has no entries.
        let attached = GasLeftBounds::default().attach(calls());
        assert!(attached.iter().all(|call| call.gas_extra.is_none()));
    }

    #[test]
    fn gas_extra_is_on_the_wire_only_when_present() {
        for gas_extra in [Some(500_000), None] {
            let call = BaseSimCallResult { inner: sim_call(true), gas_extra };
            let json = serde_json::to_value(&call).unwrap();
            assert_eq!(
                (&json["gasUsed"], &json["maxUsedGas"]),
                (&"0xc350".into(), &"0xea60".into())
            );
            match gas_extra {
                Some(_) => assert_eq!(json["gasExtra"], "0x7a120"),
                None => assert!(json.get("gasExtra").is_none(), "{json}"),
            }
            let read: BaseSimCallResult = serde_json::from_value(json).unwrap();
            assert_eq!((read.gas_extra, read.inner.gas_used), (gas_extra, 50_000));
        }
    }

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

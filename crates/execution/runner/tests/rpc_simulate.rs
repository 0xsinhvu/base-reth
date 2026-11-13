//! Wire checks for simulation block environments.

use std::sync::Arc;

use alloy_primitives::{B256, Bytes, U64, U256, hex};
use base_execution_chainspec::BaseChainSpec;
use base_node_runner::test_utils::TestHarness;
use base_test_utils::{Account, build_test_genesis};
use serde_json::{Value, json};

#[tokio::test]
async fn simulation_selects_forks_at_the_simulated_timestamp() -> eyre::Result<()> {
    // Azul activates before or after the old genesis + 12 fallback, respectively.
    for activation in [10, 20] {
        let mut genesis = build_test_genesis();
        genesis.config.osaka_time = Some(activation);
        genesis.config.extra_fields.insert("base".into(), json!({ "azul": activation }));
        let harness = TestHarness::builder()
            .with_chain_spec(Arc::new(BaseChainSpec::from_genesis(genesis)))
            .build()
            .await?;
        let client = harness.rpc_client()?;
        let probe = Account::Bob.address();
        // CLZ(1) returns 255 under Osaka and halts before Osaka. Also return TIMESTAMP.
        let code = "0x60011e6000524260205260406000f3";
        let call = json!({ "from": Account::Alice.address(), "to": probe, "gas": "0x186a0" });
        let timestamps = if activation == 10 { [3, 9, 10, 12] } else { [20, 21, 22, 24] };
        let first_override = if activation == 10 { Value::Null } else { json!({ "time": "0x14" }) };

        for trace_transfers in [false, true] {
            let response: Value = client.request("eth_simulateV1", json!([{
                "traceTransfers": trace_transfers,
                "blockStateCalls": [
                    { "blockOverrides": first_override, "stateOverrides": { (probe.to_string()): { "code": code } }, "calls": [call] },
                    { "blockOverrides": { "time": format!("0x{:x}", timestamps[1]) }, "calls": [call] },
                    { "blockOverrides": { "time": format!("0x{:x}", timestamps[2]) }, "calls": [call] },
                    { "calls": [call] }
                ]
            }, "latest"])).await?;

            let blocks = response.as_array().unwrap();
            assert_eq!(blocks.len(), 4);
            for (block, timestamp) in blocks.iter().zip(timestamps) {
                assert_eq!(block["timestamp"], json!(format!("0x{timestamp:x}")));
                let result = &block["calls"][0];
                if timestamp < activation {
                    assert_eq!(result["status"], "0x0", "CLZ must be disabled at {timestamp}");
                    assert!(!result["error"].is_null());
                } else {
                    assert_eq!(result["status"], "0x1", "CLZ must be enabled at {timestamp}");
                    assert_eq!(
                        result["returnData"],
                        json!(format!("0x{:064x}{timestamp:064x}", 255))
                    );
                }
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn simulation_preserves_same_second_denim_timestamps_and_overrides() -> eyre::Result<()> {
    let mut genesis = build_test_genesis();
    genesis.config.extra_fields.insert("base".into(), json!({ "denim": 3 }));
    let harness = TestHarness::builder()
        .with_chain_spec(Arc::new(BaseChainSpec::from_genesis(genesis)))
        .build()
        .await?;
    let client = harness.rpc_client()?;
    let probe = Account::Bob.address();
    // Return TIMESTAMP and BASEFEE. Explicit nonzero gas price keeps BASEFEE observable.
    let code = "0x426000524860205260406000f3";
    let call = json!({ "from": Account::Alice.address(), "to": probe, "gas": "0x186a0", "gasPrice": "0x3b9aca00" });
    let response: Value = client
        .request(
            "eth_simulateV1",
            json!([{
        "blockStateCalls": [
            { "stateOverrides": { (probe.to_string()): { "code": code } }, "calls": [call] },
            { "blockOverrides": { "baseFeePerGas": "0x2a" }, "calls": [call] },
            { "blockOverrides": { "time": "0x4" }, "calls": [call] },
            { "calls": [call] }
        ]
    }, "latest"]),
        )
        .await?;
    let blocks = response.as_array().unwrap();
    assert_eq!(blocks.len(), 4);
    for (index, (timestamp, base_fee)) in [(3, 0), (3, 42), (4, 0), (4, 0)].into_iter().enumerate()
    {
        assert_eq!(blocks[index]["number"], json!(format!("0x{:x}", index + 1)));
        assert_eq!(blocks[index]["timestamp"], json!(format!("0x{timestamp:x}")));
        assert_eq!(blocks[index]["calls"][0]["status"], "0x1");
        assert_eq!(
            blocks[index]["calls"][0]["returnData"],
            json!(format!("0x{timestamp:064x}{base_fee:064x}"))
        );
    }
    Ok(())
}

/// Returns `words` concatenated, as call return data.
fn return_data(words: [B256; 3]) -> Value {
    json!(Bytes::from(words.concat()))
}

#[tokio::test]
async fn simulate_v2_executes_under_the_chains_block_env() -> eyre::Result<()> {
    let prev_randao = B256::repeat_byte(0xab);
    let mut genesis = build_test_genesis();
    genesis.mix_hash = prev_randao;
    // Jovian EIP-1559 params (denominator 50, elasticity 6, no min base fee), without which the
    // next block's base fee cannot be derived and reads as zero.
    genesis.extra_data = Bytes::from_static(&hex!("0100000032000000060000000000000000"));
    let harness = TestHarness::builder()
        .with_chain_spec(Arc::new(BaseChainSpec::from_genesis(genesis)))
        .build()
        .await?;
    let client = harness.rpc_client()?;
    let genesis_hash = harness.latest_block().hash();
    let probe = Account::Bob.address();
    // Return PREVRANDAO, BASEFEE and BLOCKHASH(NUMBER - 1). No gas price, so the call only
    // executes under a nonzero BASEFEE because validation is off.
    let code = "0x4460005248602052600143034060405260606000f3";
    let call = json!({ "from": Account::Alice.address(), "to": probe, "gas": "0x186a0" });
    let params = |second_block_overrides: Value| {
        json!([{
            "blockStateCalls": [
                { "stateOverrides": { (probe.to_string()): { "code": code } }, "calls": [call] },
                { "blockOverrides": second_block_overrides, "calls": [call] }
            ]
        }, "latest"])
    };

    // V1 keeps the spec's placeholders and serves block 1's simulated hash.
    let v1: Value = client.request("eth_simulateV1", params(Value::Null)).await?;
    let v1 = v1.as_array().unwrap();
    let simulated_hash: B256 = serde_json::from_value(v1[0]["hash"].clone())?;
    for (block, parent_hash) in v1.iter().zip([genesis_hash, simulated_hash]) {
        assert_eq!(block["calls"][0]["status"], "0x1");
        assert_eq!(
            block["calls"][0]["returnData"],
            return_data([B256::ZERO, B256::ZERO, parent_hash])
        );
    }

    // V2 refuses block 1's hash, which no block on chain will have...
    let err = client
        .request::<_, Value>("eth_simulateV2", params(Value::Null))
        .await
        .expect_err("BLOCKHASH of a simulated block must fail");
    assert!(err.to_string().contains("BLOCKHASH(1)"), "{err}");

    // ...but serves one the caller pins, and the base block's prevRandao and the derived basefee.
    let pin = B256::repeat_byte(0xcd);
    let v2: Value =
        client.request("eth_simulateV2", params(json!({ "blockHash": { "1": pin } }))).await?;
    let v2 = v2.as_array().unwrap();
    assert_eq!(v2.len(), 2);
    for (block, parent_hash) in v2.iter().zip([genesis_hash, pin]) {
        let base_fee: U256 = serde_json::from_value(block["baseFeePerGas"].clone())?;
        assert_ne!(base_fee, U256::ZERO);
        assert_eq!(block["mixHash"], json!(prev_randao));
        assert_eq!(block["calls"][0]["status"], "0x1");
        assert_eq!(
            block["calls"][0]["returnData"],
            return_data([prev_randao, base_fee.into(), parent_hash])
        );
    }
    Ok(())
}

#[tokio::test]
async fn simulate_v2_reports_a_gasleft_checks_bound_as_gas_extra() -> eyre::Result<()> {
    let harness = TestHarness::new().await?;
    let client = harness.rpc_client()?;
    let hook = Account::Bob.address();
    // `if (gasleft() < 500000) revert InsufficientGas();` (`PUSH3 500000 GAS LT PUSH2 JUMPI` to
    // the revert, else STOP).
    let code = "0x6207a1205a1061000b57005b6307099c5360e21b5f5260045ffd";

    for trace_transfers in [false, true] {
        // The one call's result under `method` when it carries `gas`.
        let simulate = async |method: &'static str, gas: u64| -> eyre::Result<Value> {
            let call =
                json!({ "from": Account::Alice.address(), "to": hook, "gas": format!("{gas:#x}") });
            let blocks: Value = client
                .request(
                    method,
                    json!([{
                        "traceTransfers": trace_transfers,
                        "blockStateCalls": [
                            { "stateOverrides": { (hook.to_string()): { "code": code } }, "calls": [call] }
                        ]
                    }, "latest"]),
                )
                .await?;
            Ok(blocks[0]["calls"][0].clone())
        };
        let quantity = |value: &Value| -> eyre::Result<u64> {
            Ok(serde_json::from_value::<U64>(value.clone())?.to())
        };

        // Both run the check, which 1M gas clears, and report the same gas; V2 also reports the
        // check's bound as `gasExtra`.
        let v1 = simulate("eth_simulateV1", 1_000_000).await?;
        let v2 = simulate("eth_simulateV2", 1_000_000).await?;
        assert_eq!(v1["status"], "0x1", "{v1}");
        assert_eq!(v2["status"], "0x1", "{v2}");
        assert_eq!((&v2["gasUsed"], &v2["maxUsedGas"]), (&v1["gasUsed"], &v1["maxUsedGas"]));
        assert!(v1.get("gasExtra").is_none(), "{v1}");
        let (gas_used, gas_extra) = (quantity(&v2["gasUsed"])?, quantity(&v2["gasExtra"])?);
        assert_eq!(gas_extra, 500_000, "{v2}");

        // V1, which runs the check as the chain does, passes it with `gasUsed + gasExtra`, and with
        // the `2 * gasUsed + gasExtra` the engine sends.
        for limit in [gas_used + gas_extra, 2 * gas_used + gas_extra] {
            assert_eq!(simulate("eth_simulateV1", limit).await?["status"], "0x1", "{limit}");
        }
    }
    Ok(())
}

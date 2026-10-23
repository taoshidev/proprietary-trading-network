>  The standalone [vanta-cli](https://github.com/taoshidev/vanta-cli) repository is now deprecated, please follow the upgrade steps [here](#upgrading-from-the-standalone-vanta-cli)

# Vanta CLI

## Overview

**Vanta CLI** is a command-line tool that extends bittensor-cli for Vanta Network operations. It provides asset class selection, collateral management and entity management, and extends all standard bittensor-cli operations. Help information can be invoked for every command and option with `--help` option.

The CLI source lives in this repository under [`vanta_cli/`](../vanta_cli).

### Process Flow
From a high level, here is what happens to register with collateral on Vanta Network.

1. Register your hotkey with Vanta Network: `vanta subnets register`
2. Stake TAO into theta using your own hotkey: `vanta stake add`
3. Collateral deposit, which signs an extrinsic and sends the command off to the super validator. The super validator will then transfer the amount specified into the smart contract: `vanta collateral deposit`
4. (Optional) View collateral amount tracked in the contract: `vanta collateral list`

## Installation

The `vanta` command is now installed with Vanta. Follow the install steps in the [miner](miner.md), [entity miner](entity_miner.md) or [validator](validator.md) guide.

Check the install:

```bash
vanta --version
```

This prints the Vanta CLI version, the Vanta Network version and the bittensor-cli version.

## Upgrading from the standalone vanta-cli

Earlier releases installed the CLI as a separate `vanta-cli` package, which may cause conflicts with the current version of Vanta CLI. Uninstall the existing vanta-cli package with:

```bash
pip uninstall -y vanta-cli
pip install -r requirements.txt
python3 -m pip install -e .
```

Check the install:

```bash
vanta --version
```

## Commands

All commands are prefixed with `vanta`. For example: `vanta wallet list`

### Collateral Operations

#### Deposit Collateral
```bash
vanta collateral deposit [OPTIONS]
```
Deposit collateral to the Vanta Network.

**Options:**
- `--wallet-name, --name` - Name of the wallet to use for collateral (required)
- `--wallet-path` - Path to the wallet directory (default: `~/.bittensor/wallets`)
- `--hotkey, --wallet_hotkey` - Hotkey name or SS58 address of the hotkey
- `--network` - Network to connect to (default: `finney`)
- `--amount` - Amount of TAO to use for collateral (default: None)
- `--prompt/--no-prompt` - Whether to prompt for confirmation

#### List Collateral Balance
```bash
vanta collateral list [OPTIONS]
```
Check collateral balance for a miner address.

**Options:**
- `--wallet-name, --name` - Name of the wallet to use for collateral (required)
- `--wallet-path` - Path to the wallet directory (default: `~/.bittensor/wallets`)
- `--hotkey, --wallet_hotkey` - Hotkey name or SS58 address of the hotkey
- `--network` - Network to connect to (default: `finney`)

#### Withdraw Collateral
```bash
vanta collateral withdraw [OPTIONS]
```
Withdraw collateral from the Vanta Network.

**Options:**
- `--wallet-name, --name` - Name of the wallet (for display purposes)
- `--wallet-path` - Path to wallet directory (default: `~/.bittensor/wallets`)
- `--hotkey, --wallet_hotkey` - Hotkey name or SS58 address of the hotkey
- `--amount` - Amount to withdraw from collateral (required)
- `--prompt/--no-prompt` - Whether to prompt for confirmation

### Asset Class Selection

#### Select Asset Class
```bash
vanta asset select [OPTIONS]
```
Select the asset class your miner trades. Once selected, it cannot be changed. See the [miner guide](miner.md) for details.

**Options:**
- `--wallet-name, --name` - Name of the wallet (required)
- `--wallet-path` - Path to the wallet directory (default: `~/.bittensor/wallets`)
- `--hotkey, --wallet_hotkey` - Hotkey name or SS58 address of the hotkey
- `--network` - Network to connect to (default: `finney`)
- `--prompt/--no-prompt` - Whether to prompt for confirmation

### Entity Management

These commands are for entity miners. See the [entity miner guide](entity_miner.md) for the full workflow.

#### Register Entity
```bash
vanta entity register [OPTIONS]
```
Register your hotkey as an entity on the Vanta Network.

#### Create Subaccount
```bash
vanta entity create-subaccount [OPTIONS]
```
Create a new subaccount for your entity.

**Additional options:**
- `--account-size` - Account size in USD (prompted if omitted)
- `--asset-class` - Asset class of the subaccount (prompted if omitted)

#### Get API Key
```bash
vanta entity apikey [OPTIONS]
```
Request or retrieve the API key your entity miner uses to connect to validators.

All entity commands accept `--wallet-name`, `--wallet-path`, `--hotkey` and `--network` as above. `register` and `create-subaccount` also accept `--prompt/--no-prompt`.

## License
The MIT License (MIT)

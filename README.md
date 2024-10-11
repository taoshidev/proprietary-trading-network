# Miner Dashboard

Welcome to your PTN Miner Dashboard. This is a React application that requires a URL to be added to a `.env` file to
fetch data from your miner.

## Installation

Follow these steps to install and run the application on your local machine.

### Prerequisites

- Your PTN Miner
- A package manager like npm, yarn, or pnpm,

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/taoshidev/miner-dashboard.git
```

Navigate into the project directory:

```bash
cd miner-dashboard
```

### 2. Install Dependencies

Install the necessary dependencies by running:

```bash
npm install
```

if you are using yarn:

```bash
yarn install
```

or if you are using pnpm:

```bash
pnpm install
```

### 3. Connecting Miner-Dashboard to your Miner

#### Running your miner-dashboard on the same machine as your miner:

If you are running your miner-dashboard on the same machine as your miner, continue to section 4 [Configure Environment Variables](#4-configure-environment-variables) below.

#### Running your miner-dashboard on a different machine from your miner:

If you are running your miner-dashboard on a separate machine from your miner, you will need to add the address of the machine running the dashboard to the
allowed origins of the miner using the miner's `--dashboard_origin` flag, and enable ssh tunneling from the machine running the miner-dashboard to the machine running the miner.

1. Get the public IPv4 address of the machine running the miner-dashboard. ex: `1.1.1.1`
2. Restart the miner, adding `--dashboard_origin 1.1.1.1`
3. Note the miner URL on startup as shown in section 4. ex: `Uvicorn running on http://127.0.0.1:41511`
3. Open a SSH tunnel from the machine running the miner-dashboard
```bash
ssh -L [local_port]:localhost:[remote_port] [miner]@[address]
```
ex:
```bash
ssh -L 41511:localhost:41511 taoshi-miner@123.45.67.89
```
This maps port 41511 on your local machine running the miner-dashboard to port 41511 on the VM where the miner is running.
You are now able to connect to the miner through the miner-dashboard.

### 4. Configure Environment Variables

Add your miner URL to the environment variables. You can find your miner URL in your miner logs. Note: if a port is
occupied the miner will attempt to connect to the next available port, so confirm your port in the logs!

For example:
![Imgur](https://i.imgur.com/KusnPFt.png)

Copy the .env.example file to .env using the following command:

```bash
cp .env.example .env
```

Update the .env file

```env
VITE_MINER_URL=your-miner-url-here
```

Replace your-miner-url-here with the actual URL of your miner's uvicorn server, e.g. `http://127.0.0.1:41511`.

Note: If you opted to use a different local port than `41511` when opening a SSH tunnel above, be sure to use the local port you selected instead.

### 5. Start the Application

To start the development server, run:

```bash
npm run dev
```

if you are using yarn:

```bash
yarn dev
```

or if you are using pnpm:

```bash
pnpm dev
```

This will start the app on http://localhost:5173/ by default. Open your browser and navigate to this URL to view your
miner dashboard.

#### For Productions

To build a production ready Dashboard, run:

```bash
npm run build
```

if you are using yarn:

```bash
yarn build
```

or if you are using pnpm:

```bash
pnpm build
```


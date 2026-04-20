# Slack Notifications

## Step 1: Set up the Slack Webhook (One-time setup)

1. Open **[Slack API: Your Apps](https://api.slack.com/apps)** and sign in.
2. Click **Create New App** → **From scratch**.
3. Enter an app name (e.g. **Job Notifier**), pick the **workspace**, then create the app.
4. In the left sidebar, open **Incoming Webhooks**.
5. Turn **Activate Incoming Webhooks** **On**.
6. Click **Add New Webhook to Workspace** (at the bottom of that page).
7. Choose the **channel** where messages should appear, then allow the app.
8. Copy the **Webhook URL**. It looks like: `https://hooks.slack.com/services/T…/B…/…`

## Step 2: Use the `send_slack_message.sh` script

1. Set the `SLACK_HOOK_URI` environment variable to your Slack incoming webhook URL, optionally in your `.bashrc` or `.zshrc`, or in your SLURM script.
2. Run the `send_slack_message.sh` script with your message.

```bash
export SLACK_HOOK_URI='https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXX'
./send_slack_message.sh "Hello, world!"
```

## Step 3: Submit the Slurm demo job

The demo script sends one Slack message as soon as the job starts, then keeps the allocation alive.

```bash
cd /path/to/slack_notifications
export SLACK_HOOK_URI='https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXX'
sbatch demo_slack_notify_job.slurm
```

Notes:
- You will have to change the path to the `send_slack_message.sh` script in the SLURM script.
- The script requests the `pvc` partition for 15 minutes --- you can change them to meet your needs.
- `SLACK_HOOK_URI` should be exported in the same shell where you run `sbatch` so Slurm can pass it into the job environment.

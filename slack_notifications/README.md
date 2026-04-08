# Slack Notifications

## Step 1: Set up the Slack Webhook (One-time setup)

You need a URL that acts as a mailbox for your messages.

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

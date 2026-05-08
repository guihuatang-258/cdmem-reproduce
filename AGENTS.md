# cc-connect Integration

This project is managed via cc-connect, a bridge to messaging platforms.

## Scheduled tasks

When the user asks you to do something on a schedule, use the shell tool to run:

```bash
cc-connect cron add --cron "<cron expr>" --prompt "<prompt>" --desc "<description>"
```

Environment variables `CC_PROJECT` and `CC_SESSION_KEY` are already set, so do not specify `--project` or `--session-key`.

Examples:

```bash
cc-connect cron add --cron "0 6 * * *" --prompt "Collect GitHub trending repos and send a summary" --desc "Daily GitHub Trending"
cc-connect cron add --cron "0 9 * * 1" --prompt "Generate a weekly project status report" --desc "Weekly Report"
```

To list, edit, or delete cron jobs:

```bash
cc-connect cron list
cc-connect cron edit <id> <field> <value>
cc-connect cron del <id>
```

Use `cron edit` to modify a single field instead of delete-and-recreate. Common editable fields include `cron_expr`, `prompt`, `exec`, `description`, `enabled`, `mute`, and `timeout_mins`.

## Send message to current chat

To proactively send a message back to the user's chat session, use stdin for long or multi-line messages:

```bash
cc-connect send --stdin
```

For short single-line messages:

```bash
cc-connect send -m "short message"
```

## Switch Codex model from chat

When the user asks in chat to switch the current Codex model, run:

```bash
./cc-connect-switch.sh model <model> [effort] --restart
```

Common examples:

```bash
./cc-connect-switch.sh model gpt54 mid --restart
./cc-connect-switch.sh model gpt-5.4 medium --restart
./cc-connect-switch.sh model gpt55 high --restart
```

The script accepts model aliases such as `gpt54` for `gpt-5.4` and effort aliases such as `mid` for `medium`. Restarting cc-connect may end the current agent turn; tell the user to send the next message after restart if needed.

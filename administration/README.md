# Administration Notes

## Listing Users in an Allocation

On Stampede3, `sshare` and `sacctmgr` may only show the subset visible from your user context. A reliable way to list associated users for an allocation is:

```bash
scontrol -o show assoc_mgr account=<allocation_name> \
| awk '{for(i=1;i<=NF;i++){if($i ~ /^Account=<allocation_name>$/){acct=1} if(acct && $i ~ /^UserName=/){u=$i; sub(/^UserName=/,"",u); sub(/\(.*/,"",u); if(u!="") print u; acct=0}}}' \
| sort -u
```

Reusable form for any allocation:

```bash
ALLOC=<allocation_name>
scontrol -o show assoc_mgr account="$ALLOC" \
| awk -v a="$ALLOC" '{for(i=1;i<=NF;i++){if($i=="Account="a){acct=1} if(acct && $i ~ /^UserName=/){u=$i; sub(/^UserName=/,"",u); sub(/\(.*/,"",u); if(u!="") print u; acct=0}}}' \
| sort -u
```

Helper script in this repo (defaults to `idev_project` in `~/.idevrc`, override with `-A`):

```bash
./scripts/list_alloc_users.sh
./scripts/list_alloc_users.sh -A <allocation_name>
```

To show per-user usage for an allocation (sorted highest first):

```bash
ALLOC=<allocation_name>
scontrol -o show assoc_mgr account="$ALLOC" \
| awk -v a="$ALLOC" 'match($0,/Account=([^ ]+)/,am) && am[1]==a && match($0,/UserName=([^ (]+)/,um) && match($0,/UsageRaw\/Norm\/Efctv=([^\/]+)/,rm) {print um[1] "|" rm[1]}' \
| sort -t"|" -k2,2nr
```

Usage helper script in this repo:

```bash
./scripts/list_alloc_usage.sh
./scripts/list_alloc_usage.sh -A <allocation_name>
```

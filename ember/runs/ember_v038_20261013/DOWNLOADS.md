# Download counts and privacy

Hugging Face counts file requests over a rolling 30-day window, not unique users.
This native package includes an accurate config.json for model download counting.
Internal tests use local artifacts/cache. Necessary online downloads should be
logged with agent, purpose and pinned revision. Upload verification uses metadata
and blob/LFS hashes without downloading weights. Historical counters cannot
reliably separate internal and external use; there is no user tracking in this
package. See the project tools/hf_download_audit.py for request accounting.

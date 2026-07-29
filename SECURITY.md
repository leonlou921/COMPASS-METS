# Security

Do not open an issue containing patient data, credentials, private challenge
assets, checkpoints, or prediction files. Report a source-code security issue
privately to the repository maintainers through GitHub's security advisory
interface.

The training and inference scripts do not require application-level network
access after dependencies and authorized data are available. Keep all private
artifacts in external, access-controlled storage and pass their paths through
environment variables.

# Security

Do not open an issue containing patient data, credentials, private challenge
assets, checkpoints, or prediction files. Report a source-code security issue
privately to the repository maintainers through GitHub's security advisory
interface.

The inference image is designed to run offline. It does not require network
access and writes only to the mounted output directory and its temporary work
directory.

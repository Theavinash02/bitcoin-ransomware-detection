---
title: Bitcoin Ransomware Detector
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

Scores one BitcoinHeist address-day for ransomware risk with a LightGBM model trained on data
before July 2016. Built from https://github.com/theavinash02/bitcoin-ransomware-detection; run
`ransomdet export-demo` there to refresh `artifacts/` and `ransomdet/` before pushing this folder.

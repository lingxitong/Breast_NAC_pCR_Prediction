# Third-party: MambaMIL

本目录为 [isyangshu/MambaMIL](https://github.com/isyangshu/MambaMIL)（MICCAI 2024, arXiv:2403.06800）的 vendored 副本，供本项目的 `mamba_mil` / `trans_mil` / `s4model` 使用。

- 上游许可见本目录 `README.md` / `mamba/LICENSE`（学术非商业用途等条款以原仓库为准）
- 引用：

```text
@article{yang2024mambamil,
  title={MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology},
  author={Yang, Shu and Wang, Yihui and Chen, Hao},
  journal={arXiv preprint arXiv:2403.06800},
  year={2024}
}
```

本项目仅使用其中的 `models/`（MIL 网络）与 `mamba/`（含 SRMamba/BiMamba 的 mamba_ssm）。
训练脚本、划分与数据 CSV 仍保留自上游，日常训练请使用上级目录的 `main_pcr_train.py`。

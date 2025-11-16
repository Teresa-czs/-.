
---


# FAR-Trans 研究版

本项目为 **FAR-Trans** 模型的研究版，用于用户与资产偏好预测与矩阵导出。

## 数据说明
- 使用的数据文件：
  - `customer_information.csv` — 客户基础信息  
  - `assets_latest.csv` — 最新资产信息  
  - `customers_latest.csv` — 最新客户快照  
- 输出文件：
  - `user_asset_preferences_full.csv` — 全量用户×资产偏好矩阵  
  - 模型权重：`xdeepfm_best.pt`

## 使用步骤
1. **加载数据**
   ```python
   customers = pd.read_csv(FILES['customers'])
   assets = pd.read_csv(FILES['assets'])
   ```
2. **训练模型**
   - 模型：xDeepFM（含 CIN 层）
   - 最优模型保存在 `xdeepfm_best.pt`
3. **验证评估**
   - 自动生成 ROC / PR 图、分类指标等
   - 输出：`xdeepfm_ROC.png`、`xdeepfm_PR.png`、`xdeepfm_cls_report.csv`
4. **导出偏好矩阵**
   ```text
   [Step4B 研究版] 宽矩阵已导出到：user_asset_preferences_full.csv
   共写入用户 29,090，矩阵维度约 (29,090, 807)
   ```

## 环境要求
Python 3.9+、PyTorch、pandas、numpy。

## 备注
- 自动支持 GPU/CPU 加速。  
- 数据需无缺失值。  
- 本仓库仅用于研究验证。  

---


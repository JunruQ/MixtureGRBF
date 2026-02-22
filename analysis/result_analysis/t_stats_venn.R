library(readr)           # 读入CSV
library(dplyr)           # 数据处理
library(ggplot2)         # ggplot支持

# 读入数据
t_stat_path <- 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/t_stats_protein_linear_reg.csv'
t_stats <- read_csv(t_stat_path, show_col_types = FALSE)

# 转换 Subtype 为整数类型
t_stats$Subtype <- as.integer(t_stats$Subtype)

# 设置 top_k
top_k <- 100

# 初始化一个 list 用于存储每个 Subtype 的 Top biomarker
biomarker_list <- list()

# 按子类型循环，筛选 top_k 个 |t| 最大的 Biomarker
for (i in 1:max(t_stats$Subtype)) {
  top_biomarkers <- t_stats %>%
    filter(Subtype == i) %>%
    arrange(desc(abs(t))) %>%
    slice_head(n = top_k) %>%
    pull(Biomarker)
  
  biomarker_list[[paste0("Subtype ", i)]] <- unique(top_biomarkers)
}

# 自定义颜色
subtype_colors <- c(
  '#0072BD',  # Subtype 1
  '#EDB120',  # Subtype 2
  '#77AC30',  # Subtype 3
  '#D95319',  # Subtype 4
  '#A2142F'   # Subtype 5
)

# 使用 ggVennDiagram 绘制维恩图
p <- ggVennDiagram(biomarker_list, label_alpha = 0) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(legend.position = "none") +
  ggtitle(paste("Top", top_k, "Biomarkers by |t| Value across Subtypes"))

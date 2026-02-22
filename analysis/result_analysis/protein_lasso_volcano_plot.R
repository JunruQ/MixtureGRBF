# 加载必要的包
library(data.table)
library(glmnet)
library(ggplot2)
library(dplyr)
library(ggrepel) # 用于智能放置文本标签

df <- fread("input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv")
y <- df$stage
X <- as.matrix(df[, 8:ncol(df)])
feature_names <- colnames(df)[8:ncol(df)]

# --- 3. 模型拟合与系数提取 ---

# 设置并拟合 Lasso 模型 (与您的代码相同)
optimal_alpha <- 0.01174388887841015
# 注意：在 glmnet 中，lambda 是惩罚项的强度。您提供的 optimal_alpha 看起来更像一个 lambda 值。
# 这里我们将其用作 lambda。alpha=1 表示这是 Lasso 回归。
lasso_model <- glmnet(X, y, alpha = 1, lambda = optimal_alpha, standardize = FALSE)

# 提取非零系数的特征
nonzero_indices <- which(lasso_model$beta != 0)
nonzero_features <- feature_names[nonzero_indices]
X_nonzero <- X[, nonzero_indices, drop = FALSE]
nonzero_coefs <- as.vector(lasso_model$beta[nonzero_indices])

# --- 4. 计算 P 值 ---

# 使用线性回归计算 p 值
if (length(nonzero_features) > 0) {
    ols_data <- as.data.frame(cbind(y = y, X_nonzero))
    ols_model <- lm(y ~ ., data = ols_data)
    p_values <- summary(ols_model)$coefficients[-1, 4] # 排除截距的 p 值
    log_pval <- -log10(p_values)
} else {
    # 如果没有非零系数，则创建空数据框以避免错误
    p_values <- numeric(0)
    log_pval <- numeric(0)
}


# --- 5. 创建用于绘图的数据框 ---

# 定义现代感的颜色
modern_red <- "#c0392b"  # 一种柔和的深红色
modern_blue <- "#2980b9" # 一种平静的蓝色
grey_color <- "grey50"

plot_data <- data.frame(
  coef = nonzero_coefs,
  log_pval = log_pval,
  feature = nonzero_features,
  p_value = p_values,
  abs_coef = abs(nonzero_coefs)
)

# 添加颜色和显著性分组
plot_data <- plot_data %>%
  mutate(
    significance = ifelse(p_value < 0.05, "Significant", "Not Significant"),
    color_group = case_when(
      p_value < 0.05 & coef > 0  ~ modern_red,
      p_value < 0.05 & coef < 0  ~ modern_blue,
      TRUE                       ~ grey_color
    )
  )

# 选出要标记的最显著的特征 (例如，p 值最小的前 10 个)
top_n <- 5
features_to_label <- plot_data %>% 
  arrange(coef) %>% 
  group_by(coef < 0) %>% 
  slice_max(log_pval, n = top_n) %>% 
  ungroup() %>% 
  distinct()


# --- 6. 生成美化后的火山图 ---
p <- ggplot(plot_data, aes(x = coef, y = log_pval)) +
  geom_point(
    aes(fill = coef, size = log_pval),
    shape = 21,              # 圆形，允许填充和描边分开控制
    color = "black",         # 固定黑色描边
    stroke = 0.2,            # 描边线宽
    alpha = 0.7
  ) +
  scale_fill_gradient2(      # 渐变填充色：红白蓝（效果方向）
    low = modern_blue,         # 蓝色：负效应
    mid = grey_color,           # 中性
    high = modern_red,        # 红色：正效应
    midpoint = 0
  ) +
  # scale_color_identity() +
  scale_size_continuous(range = c(0, 4), guide = "none") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey30") +
  # geom_vline(xintercept = 0, linetype = "solid", color = "grey30") +
  geom_text_repel(
    data = features_to_label,
    aes(label = feature),
    size = 3, # 9pt ≈ 3mm in ggplot2
    family = "Arial", # 设置为Arial
    color = "black",
    box.padding = 0.2,
    point.padding = 0.1,
    max.overlaps = Inf,
    segment.color = 'grey50',
    force = 1,
    ylim = c(20, NA),
    max.iter = 10000     # 增加迭代次数
  ) +
  labs(
    x = "Lasso coefficient (effect size)",
    y = "-log10(p-value)",
    title = "Volcano plot of protein contributions"
  ) +
  theme_bw(base_family = "Arial") + # 设置全局字体为Arial
  theme(
    plot.title = element_text(size = 12, hjust = 0.5), # 9pt标题，粗体
    axis.title = element_text(size = 12, color = "black"),
    axis.text = element_text(size = 12, color = "black"), # 9pt轴刻度文字，粗体
    # axis.line = element_line(color = "black"),      # 坐标轴线为黑色
    axis.ticks = element_line(color = "black", linewidth = 0.25),     # 刻度线为黑色
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    # panel.grid.major = element_line(color = "grey90", linetype = "dashed"),
    # panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.margin = margin(t = 0.05, r = 1, b = 0.05, l = 0.05, unit = "cm") # 增加右侧padding
  )

# 打印图形
print(p)

# --- 7. 保存图形 ---
ggsave("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/lasso_protein_volcano_plot.png", 
       plot = p, 
       dpi = 300, 
       width = 7.5, 
       height = 7, 
       units = "cm") # 设置尺寸为8cm×8cm

cat("Volcano plot has been generated and saved as 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/lasso_protein_volcano_plot.png'\n")
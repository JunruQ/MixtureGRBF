# 加载必要的包
library(data.table)
library(glmnet)
library(ggplot2)
library(dplyr)
library(scales) # 用于z-score标准化

# --- 定义颜色 ---
modern_red <- "#c0392b"  # 柔和的深红色
modern_blue <- "#2980b9" # 平静的蓝色

# --- 数据加载 ---
df <- fread("input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv")
y <- df$stage
X <- as.matrix(df[, 8:ncol(df)])
feature_names <- colnames(df)[8:ncol(df)]

# --- Lasso模型拟合 ---
optimal_alpha <- 0.01174388887841015
lasso_model <- glmnet(X, y, alpha = 1, lambda = optimal_alpha, standardize = FALSE)

# --- 预测stage ---
pred_stage <- predict(lasso_model, newx = X, s = optimal_alpha, type = "response")
pred_stage <- as.vector(pred_stage)

# --- 计算z-scored age gap ---
age_gap = pred_stage - y
zscored_age_gap <- age_gap

# --- 创建绘图数据框 ---
plot_data <- data.frame(
  chronological_stage = y,
  predicted_stage = pred_stage,
  zscored_age_gap = zscored_age_gap
)

r_val <- cor(plot_data$chronological_stage, plot_data$predicted_stage, method = "pearson")
label_text <- sprintf("r = %.2f", r_val)

# --- 生成散点图 ---
p <- ggplot(plot_data, aes(x = chronological_stage, y = predicted_stage)) + 
  annotate("text",
           x = min(plot_data$chronological_stage, na.rm = TRUE),
           y = max(plot_data$predicted_stage, na.rm = TRUE),
           label = label_text,
           hjust = 0, vjust = 1,
           size = 4.5) +
  # 散点，添加jitter，按z-scored age gap着色
  geom_jitter(aes(color = zscored_age_gap), size = 2, alpha = 0.7, width = 0.5, height = 0) +
  # 蓝色-白色-红色渐变
  scale_color_gradient2(
    low = modern_blue, mid = "white", high = modern_red,
    midpoint = 0,
    name = "Age gap",
    guide = guide_colorbar(
        title.position = "right",  # 放在 colorbar 的右侧
        title.theme = element_text(angle = 90, size = 12, hjust = 0.5),
        barheight = unit(0.78, "npc"),  # 与坐标轴等高
        barwidth = unit(0.4, "cm")   # 可调整
    )
  ) +
  # 黑色趋势线
  # geom_smooth(method = "lm", color = "black", se = FALSE, linetype = "solid") +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "solid") +
  # 设置标签
  labs(
    x = "Chronological age (years)",
    y = "Predicted age (years)",
    title = "Plasma protein age prediction"
  ) +
  # 使用Arial字体并设置样式
  theme_bw(base_family = "Arial") +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5), # 9pt标题，粗体
    axis.title = element_text(size = 12, color = "black"),
    axis.text = element_text(size = 12, color = "black"), # 9pt轴刻度文字，粗体
    # axis.line = element_line(color = "black"),      # 坐标轴线为黑色
    axis.ticks = element_line(color = "black", linewidth = 0.25),     # 刻度线为黑色
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    legend.position = "right", # Colorbar在右侧
    # legend.key.height = unit(6, "cm"), # Colorbar与图形等高（略小于8cm以适应标题）
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 12),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, unit = "cm") # 无padding
  )

# 打印图形
print(p)

# --- 保存图形 ---
ggsave("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/lasso_stage_plot.png", 
       plot = p, 
       dpi = 300, 
       width = 9, 
       height = 7, 
       units = "cm")

cat("Scatter plot of predicted vs chronological stage with jitter has been generated and saved as 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/lasso_stage_plot.png'\n")

# # 加载必要的包
# library(data.table)
# library(glmnet)
# library(ggplot2)
# library(dplyr)

# # --- 定义颜色 ---
# # grey_color <- "grey50"  # 统一灰色
# # grey_color <- "#a4cde1"
# grey_color <- "#a19ec1"

# # --- 数据加载 ---
# df <- fread("input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv")
# y <- df$stage
# X <- as.matrix(df[, 8:ncol(df)])
# feature_names <- colnames(df)[8:ncol(df)]

# # --- Lasso模型拟合 ---
# optimal_alpha <- 0.01174388887841015
# lasso_model <- glmnet(X, y, alpha = 1, lambda = optimal_alpha, standardize = FALSE)

# # --- 预测stage ---
# pred_stage <- predict(lasso_model, newx = X, s = optimal_alpha, type = "response")
# pred_stage <- as.vector(pred_stage)

# # --- 创建绘图数据框 ---
# plot_data <- data.frame(
#   chronological_stage = y,
#   predicted_stage = pred_stage
# )

# r_val <- cor(plot_data$chronological_stage, plot_data$predicted_stage, method = "pearson")
# label_text <- sprintf("r = %.2f", r_val)

# # --- 生成散点图 ---
# p <- ggplot(plot_data, aes(x = chronological_stage, y = predicted_stage)) + 
#   annotate("text",
#            x = min(plot_data$chronological_stage, na.rm = TRUE),
#            y = max(plot_data$predicted_stage, na.rm = TRUE),
#            label = label_text,
#            hjust = 0, vjust = 1,
#            size = 3,
#            fontface = "bold") +
#   # 散点，统一灰色，添加jitter
#   geom_jitter(color = grey_color, size = 0.5, alpha = 0.1, width = 0.5, height = 0) +
#   # y=x线
#   geom_abline(intercept = 0, slope = 1, color = "#cd3031", linetype = "solid") +
#   # 设置标签
#   labs(
#     x = "Chronological age (years)",
#     y = "Predicted age (years)",
#     title = "Plasma protein age prediction"
#   ) +
#   # 使用Arial字体并设置样式
#   theme_bw(base_family = "Arial") +
#   theme(
#     plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
#     axis.title = element_text(size = 12, face = "bold", color = "black"),
#     axis.text = element_text(size = 12, face = "bold", color = "black"),
#     axis.ticks = element_line(color = "black", linewidth = 0.25),
#     panel.grid = element_blank(),
#     panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
#     legend.position = "none", # 移除图例
#     plot.margin = margin(0.1, 0.1, 0.1, 0.1, unit = "cm")
#   )

# # 打印图形
# print(p)

# # --- 保存图形 ---
# ggsave("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/lasso_stage_plot.png", 
#        plot = p, 
#        dpi = 300, 
#        width = 6.5, 
#        height = 7, 
#        units = "cm")

# cat("Scatter plot of predicted vs chronological stage with jitter has been generated and saved as 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/lasso_stage_plot.png'\n")
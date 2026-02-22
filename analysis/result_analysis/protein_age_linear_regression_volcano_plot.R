# 加载必要的包
library(data.table)
library(ggplot2)
library(dplyr)
library(ggrepel) # 用于智能放置文本标签

df <- fread("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/protein_age_linear_regression_results.csv")

# 获取唯一的subtype
subtypes <- unique(df$subtype)
top_n <- 5

# 定义现代感的颜色
modern_red <- "#c0392b"  # 一种柔和的深红色
modern_blue <- "#2980b9" # 一种平静的蓝色
grey_color <- "grey50"

# 为每个subtype绘制火山图
for (sub in subtypes) {
  # 子集数据
  sub_df <- df[df$subtype == sub, ]
  
  # 计算log_pval
  sub_df$log_pval <- -log10(sub_df$p_corrected)
  
  # 添加颜色和显著性分组
  plot_data <- sub_df %>%
    mutate(
      significance = ifelse(pvalue < 0.05, "Significant", "Not Significant"),
      color_group = case_when(
        pvalue < 0.05 & beta > 0  ~ modern_red,
        pvalue < 0.05 & beta < 0  ~ modern_blue,
        TRUE                       ~ grey_color
      )
    )
  
  # 选出要标记的最显著的特征 (例如，p 值最小的前 top_n 个，按正负beta分组)
  features_to_label <- plot_data %>% 
    arrange(beta) %>% 
    group_by(beta < 0) %>% 
    slice_max(log_pval, n = top_n) %>% 
    ungroup() %>% 
    distinct()
  
  # --- 生成美化后的火山图 ---
  p <- ggplot(plot_data, aes(x = beta, y = log_pval)) +
    geom_point(
      aes(fill = beta, size = log_pval),
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
      aes(label = features_to_label$protein),
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
      title = paste("Volcano plot of protein contributions - Subtype", sub)
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
  
  # --- 保存图形 ---
  output_file <- paste0("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/protein_age_linear_regression_volcano_plot_subtype_", sub, ".png")
  ggsave(output_file, 
         plot = p, 
         dpi = 300, 
         width = 7.5, 
         height = 7, 
         units = "cm") # 设置尺寸为8cm×8cm
  
  cat("Volcano plot for Subtype", sub, "has been generated and saved as '", output_file, "'\n")
}
library(ggplot2)
library(dplyr)
library(stringr)
# 如果需要加载字体，请取消下面两行的注释
# library(extrafont)
# loadfonts(device = "pdf") # 或 "win"

# 读取数据
df <- read.csv("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/gsea_results.csv")

# 1. 数据预处理
top_n_terms <- df %>%
  group_by(Subtype) %>%
  arrange(p.adjust) %>%
  slice_head(n = 10) %>%
  ungroup() %>%
  mutate(logp = -log10(p.adjust),
         signed_logp = ifelse(.sign == "activated", logp, -logp)) %>%
  group_by(Subtype) %>%
  arrange(desc(signed_logp)) %>%
  # 保证每个 Subtype 内部的排序
  mutate(Description = factor(Description, levels = rev(unique(Description)))) %>%
  ungroup()

# 计算对称颜色范围
max_abs <- max(abs(top_n_terms$signed_logp), na.rm = TRUE)

# 2. 绘图
p <- ggplot(top_n_terms, aes(x = GeneRatio, y = Description)) +
  geom_point(aes(fill = signed_logp, size = GeneRatio),
             shape = 21, colour = "black", stroke = 0.6) +
  
  # 分面
  facet_grid(Subtype ~ ., scales = "free_y", space = "free_y") +
  
  # 颜色设置
  scale_fill_gradient2(
    low = "#2166ac",      
    mid = "#f7f7f7",      
    high = "#d7191c",     
    midpoint = 0,
    limits = c(-max_abs, max_abs),
    name = "Signed -log10(p.adjust)"
  ) +
  
  scale_size_continuous(range = c(3, 8), name = "Gene Ratio") +
  scale_y_discrete(labels = function(x) str_wrap(x, width = 50)) +
  
  # 主题美化：移除标题并设置 Arial 字体
  theme_bw() +
  labs(x = "Gene Ratio", y = "Pathway Description") + # 删除了 title 和 subtitle
  theme(
    # 全局字体设置为 Arial
    text = element_text(family = "Arial"),
    strip.text.y = element_text(angle = 0, face = "bold", size = 12),
    axis.text.x = element_text(size = 9),
    axis.text.y = element_text(size = 9),
    panel.grid.major.x = element_line(linetype = "dashed", colour = "grey80"),
    legend.position = "right"
  )

# 显示图形
print(p)

# 3. 保存图片：单位改为 cm
# 宽度 28cm, 高度 40cm 左右比较适合 50 条左右的通路展示
ggsave("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/gsea_dotplot.png", 
       width = 15, 
       height = 20, 
       units = "cm", 
       dpi = 300)
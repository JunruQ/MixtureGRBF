# 加载必要的包
library(ggplot2)
library(data.table)
library(cowplot)

# 设置主题（Arial Bold, 9pt）
theme_set(theme_minimal(base_family = "Arial", base_size = 9) + 
            theme(axis.title = element_text(size = 9),
                  axis.text = element_text(size = 9),
                  plot.title = element_text(size = 9),
                  legend.title = element_text(size = 9),
                  legend.text = element_text(size = 9)))

# 参数设置
nsubtype <- 5
exp_name <- 'ukb_MixtureGRBF_cv_nsubtype_biom17'
input_table_path <- 'input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv'
subtype_order_path <- sprintf('output/result_analysis/%s/%d_subtypes/all_cause_mortality_order.csv', exp_name, nsubtype)
output_dir <- sprintf('output/result_analysis/%s/%d_subtypes', exp_name, nsubtype)

star_proteins <- c('SCARF2', 'LRFN2', 'PODXL2', 'FBLN2')
star_colors <- c(SCARF2 = "#ff7189", LRFN2 = "#eee8a9", PODXL2 = "#6bbb5c", FBLN2 = "#00b3ff")

# 读取数据
df <- fread(input_table_path)
subtype_order <- fread(subtype_order_path, header = FALSE)[[1]]
biomarker_names <- names(df)[8:ncol(df)]

# 存储子图
plot_list <- list()

for (i in seq_len(nsubtype)) {
  k <- subtype_order[i]
  trajectory_path <- sprintf('output/%s/%d_subtypes/trajectory%d.csv', exp_name, nsubtype, k)
  if (!file.exists(trajectory_path)) {
    warning(sprintf("Missing file: %s", trajectory_path))
    next
  }

  traj <- fread(trajectory_path)
  traj$Age <- 39:70
  traj_long <- melt(traj, id.vars = "Age", variable.name = "Protein", value.name = "Value")

  traj_star <- traj_long[Protein %in% star_proteins]
  traj_others <- traj_long[!Protein %in% star_proteins]

  p <- ggplot() +
    geom_line(data = traj_others, aes(x = Age, y = Value, group = Protein),
              color = "gray80", size = 0.3, alpha = 0.5) +
    geom_line(data = traj_star, aes(x = Age, y = Value, color = Protein),
              size = 0.8, alpha = 1) +
    scale_color_manual(values = star_colors, name = "Star Proteins") +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.4) +
    labs(title = paste0("Subtype ", i), x = "Age", y = "Protein Value") +
    ylim(-2.2, 2.2) +
    theme(legend.position = "none")  # 图中不显示图例

  plot_list[[i]] <- p
}

# --- 创建单独的图例子图 ---
legend_plot <- ggplot(data.frame(Age = 1:2, Protein = star_proteins), 
                      aes(x = Age, y = Age, color = Protein)) +
  geom_line(size = 1) +
  scale_color_manual(values = star_colors, name = "Star Proteins") +
  theme_void() +
  theme(legend.position = "top",
        legend.title = element_text(size = 9, margin = margin(b = 10)),  # 标题与图例项间距
        legend.text = element_text(size = 8),
        legend.key.height = unit(0.3, "cm"),  # 图例键高度
        legend.key.width = unit(1.2, "cm"),   # 图例键宽度
        legend.box = "vertical",              # 确保标题在上方
        legend.spacing.x = unit(0.5, "cm"))   # 图例项水平间距

# 提取图例 grob
legend_grob <- get_legend(legend_plot +
                            guides(color = guide_legend(nrow = 2, ncol = 2, byrow = TRUE)))  # 2x2 图例布局

# 合并图例子图 + 所有轨迹子图
final_plot <- plot_grid(legend_grob, 
                        plot_grid(plotlist = plot_list, ncol = 1, align = "v"),
                        ncol = 1,
                        rel_heights = c(0.25, 1))  # 图例高度占比 25%

# 保存图像
ggsave(filename = file.path(output_dir, "star_protein_trajectories.png"),
       plot = final_plot,
       width = 10, height = 3 * length(plot_list) + 2, units = "cm", dpi = 300)

cat(sprintf("✅ 图像已保存至：%s/star_protein_trajectories.png\n", output_dir))
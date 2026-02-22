library(ggplot2)
library(dplyr)

# File paths
input_file <- "output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/specificity_t_stats.csv"
output_plot <- "output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/specificity_t_stats_visualization.png"

# Create output directory if it doesn't exist
dir.create(dirname(output_plot), showWarnings = FALSE, recursive = TRUE)

# Load data
cat("Loading data...\n")
data <- read.csv(input_file, stringsAsFactors = FALSE)
cat("Data dimensions:", dim(data), "\n")

# Check data
if (!all(c("subtype", "cluster_id", "t_statistic") %in% colnames(data))) {
  stop("Required columns (subtype, cluster_id, t_statistic) not found in data")
}

# Summarize data
cat("Number of unique subtypes:", length(unique(data$subtype)), "\n")
cat("Number of unique cluster_ids:", length(unique(data$cluster_id)), "\n")

# Prepare data: add absolute t_statistic
data <- data %>%
  mutate(abs_t_stat = abs(t_statistic)) %>%
  arrange(cluster_id)  # Sort cluster_id for consistent display

# Calculate max absolute t_statistic for color scaling
max_abs_t <- max(data$abs_t_stat, na.rm = TRUE)
cat("Max absolute t-statistic:", max_abs_t, "\n")

# Create breaks for Y axis (show every 10th cluster_id)
cluster_ids <- unique(data$cluster_id)
y_breaks <- cluster_ids[seq(1, length(cluster_ids), by = 10)]

# Create scatter plot
cat("Creating visualization...\n")
p <- ggplot(data, aes(x = subtype, y = factor(cluster_id, levels = cluster_ids), 
                     size = abs_t_stat, color = t_statistic)) +
  geom_point(alpha = 0.6) +
  scale_size_continuous(name = "|t-statistic|", range = c(1, 10)) +
  scale_color_gradientn(
    colors = c("#007bff", "white", "#ff5500"),
    values = scales::rescale(c(-max_abs_t, 0, max_abs_t)),
    name = "t-statistic",
    guide = guide_colorbar(order = 1)
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 4),  # Smaller font for 461 cluster_ids
    axis.title = element_text(size = 12),
    legend.position = "right",
    legend.box = "vertical",
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank()
  ) +
  scale_y_discrete(breaks = y_breaks) +
  labs(
    x = "Subtype",
    y = "Cluster ID",
    title = "T-Statistic by Subtype and Cluster"
  )

# Save plot
cat("Saving plot to", output_plot, "\n")
ggsave(output_plot, plot = p, width = 10, height = 30, dpi = 300)

cat("Visualization complete.\n")
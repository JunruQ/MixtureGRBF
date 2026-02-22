library(data.table)
library(ggplot2)
library(dplyr)
library(scales)

modern_red <- "#c0392b"
modern_blue <- "#2980b9"

blood_chem_path <- "data/ClinicalLabData.csv"
exp_name <- "ukb_MixtureGRBF_cv_nsubtype_biom17"
nsubtype <- 5
subtype_stage_path <- paste0("output/", exp_name, "/", nsubtype, "_subtypes/subtype_stage.csv")

subtype_stage <- fread(subtype_stage_path)
blood_chem_df <- fread(blood_chem_path)

setnames(subtype_stage, "PTID", "eid")

subtype_stage <- merge(subtype_stage, blood_chem_df, by = "eid", all.x = TRUE)

X <- as.matrix(subtype_stage[, !c("eid", "subtype", "stage"), with = FALSE])
y <- subtype_stage$stage

X <- apply(X, 2, function(col) {
  col[is.na(col)] <- mean(col, na.rm = TRUE)
  return(col)
})

model <- lm(y ~ X)

pred_stage <- predict(model, newdata = as.data.frame(X))
pred_stage <- as.vector(pred_stage)

age_gap <- pred_stage - y
zscored_age_gap <- age_gap

plot_data <- data.frame(
  chronological_stage = y,
  predicted_stage = pred_stage,
  zscored_age_gap = zscored_age_gap
)

r_val <- cor(plot_data$chronological_stage, plot_data$predicted_stage, method = "pearson")
label_text <- sprintf("r = %.2f", r_val)

p <- ggplot(plot_data, aes(x = chronological_stage, y = predicted_stage)) + 
  annotate("text",
           x = min(plot_data$chronological_stage, na.rm = TRUE),
           y = max(plot_data$predicted_stage, na.rm = TRUE),
           label = label_text,
           hjust = 0, vjust = 1,
           size = 4.5) +
  geom_jitter(aes(color = zscored_age_gap), size = 2, alpha = 0.7, width = 0.5, height = 0) +

  scale_color_gradient2(
    low = modern_blue, mid = "white", high = modern_red,
    midpoint = 0,
    name = "Age gap",
    guide = guide_colorbar(
      title.position = "right",
      title.theme = element_text(angle = 90, size = 12, hjust = 0.5),
      barheight = unit(0.78, "npc"),
      barwidth = unit(0.4, "cm")
    )
  ) +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "solid") +
  labs(
    x = "Chronological age (years)",
    y = "Predicted age (years)",
    title = "Blood routine age prediction"
  ) +
  theme_bw(base_family = "Arial") +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 12, color = "black"),
    axis.text = element_text(size = 12, color = "black"),
    axis.ticks = element_line(color = "black", linewidth = 0.25),
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    legend.position = "right",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 12),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, unit = "cm")
  )

output_dir <- paste0("output/result_analysis/", exp_name, "/", nsubtype, "_subtypes")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(
  paste0(output_dir, "/blood_chem_predicted_age_delta_scatter.png"),
  plot = p,
  dpi = 300,
  width = 8,
  height = 20,
  units = "cm"
)

cat("Scatter plot of predicted vs chronological stage with jitter has been generated and saved as '",
    paste0(output_dir, "/blood_chem_predicted_age_delta_scatter.png"), "'\n")
# Load required libraries
library(ggplot2)
library(RColorBrewer)  # For better color palettes

# Set the output directory (adjust this path to match your OUTPUT_DIR)
output_dir <- "output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes"

# Read the data
data <- read.csv(file.path(output_dir, "tissue_index_slope_diff.csv"))

# Ensure Subtype and Indicator are factors with correct order
# Subtype: S1, S2, ..., S5 (left to right)
# Indicator: Reverse order to match the top-to-bottom arrangement in the image
data$Subtype <- factor(data$Subtype, levels = unique(data$Subtype))
data$Indicator <- factor(data$Indicator, levels = rev(unique(data$Indicator)))

# Create a new variable for color: LogP adjusted by the sign of Beta
# Positive Beta with high LogP -> positive values (red)
# Negative Beta with high LogP -> negative values (blue)
# Low LogP -> near 0 (white)
data$SignedLogP <- data$LogP * sign(data$Beta)
data$Category <- factor(data$Category)

# Create the dot plot
p <- ggplot(data, aes(x = Subtype, y = Indicator)) +
  geom_point(aes(size = abs(Beta), color = SignedLogP)) + 
  facet_grid(Category ~ ., scales = "free_y", space = "free_y") +  # 纵向分面
  scale_size_continuous(
    range = c(0.5, 5),  # Size range for circles
    name = "Beta\nCoefficient\n(Absolute)"  # Legend title with line breaks
  ) +
  scale_color_distiller(
    palette = "RdBu",  # Use RColorBrewer's RdBu palette
    direction = -1,  # Reverse the palette: negative (blue), positive (red)
    name = "Significance\n-log10(p)",  # Colorbar title with line break
    limits = c(-max(abs(data$SignedLogP), na.rm = TRUE), max(abs(data$SignedLogP), na.rm = TRUE)),  # Symmetric limits
    oob = scales::squish  # Squish values outside the limits
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "gray", linetype = "dashed"),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(0.1, "cm"),  # Space between legend items
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8)
  ) +
  labs(x = "Subtypes", y = "Indicators") +
  theme(
    strip.text.y = element_text(angle = 0),  # 调整分面标签角度
    panel.spacing = unit(0.2, "lines")  # 调整分面间距
  )

p <- p + theme(
  axis.text.x = element_text(size = 10, colour = "black"),
  axis.text.y = element_text(size = 10, colour = "black")
)


ggsave(
  file.path(output_dir, "tissue_index_dotplot.png"), 
  plot = p, 
  width = 17, height = 15, units = "cm", dpi = 300
)

# Optionally, display the plot
print(p)
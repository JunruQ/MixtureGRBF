# 加载所需包
library(TissueEnrich)
library(SummarizedExperiment)
library(dplyr)
library(ggplot2)
library(cowplot)

# 读取数据
data <- read.csv('t_stats_by_subtype.csv')

# 对每个 subtype 进行 Tissue Enrichment 分析并可视化
subtypes <- unique(data$subtype)
enrichment_list <- list()
pvalue_plot_list <- list()
logfc_plot_list <- list()

# 对每个 subtype 进行分析
for (sub in subtypes) {
  # 筛选当前 subtype 中显著的基因
  sub_data <- data %>% 
    filter(subtype == sub, significant == "True")
  
  # 如果没有显著的基因，跳过此 subtype
  if (nrow(sub_data) == 0) {
    warning(paste("No significant genes found for subtype", sub, "- skipping"))
    next
  }
  
  # 提取显著基因
  sub_genes <- unique(as.character(sub_data$biom))
  
  # 创建 GeneSet 对象
  gs <- GeneSet(geneIds = sub_genes,
                organism = "Homo Sapiens",
                geneIdType = SymbolIdentifier())
  
  # 运行 Tissue Enrichment 分析
  output <- teEnrichment(gs)
  
  # 提取富集结果
  seEnrichmentOutput <- output[[1]]

  # 创建数据框，确保正确提取列
  enrichmentOutput <- data.frame(
    Tissue = row.names(seEnrichmentOutput),
    Log10PValue = assay(seEnrichmentOutput)[, "Log10PValue"],
    Tissue.Specific.Genes = assay(seEnrichmentOutput)[, "Tissue.Specific.Genes"],
    Fold.Change = assay(seEnrichmentOutput)[, "fold.change"],
    stringsAsFactors = FALSE
  )
  
  # 如果 Fold.Change 不存在，尝试小写或其他可能名称
  if (!"Fold.Change" %in% colnames(assay(seEnrichmentOutput))) {
    if ("fold.change" %in% colnames(assay(seEnrichmentOutput))) {
      enrichmentOutput$Fold.Change <- assay(seEnrichmentOutput)[, "fold.change"]
    } else {
      warning(paste("Fold.Change not found for subtype", sub, "- skipping Fold Change plot"))
      enrichmentOutput$Fold.Change <- NA
    }
  }
  
  # 添加 subtype 信息
  enrichmentOutput$Subtype <- sub
  
  # 存储富集结果
  enrichment_list[[as.character(sub)]] <- enrichmentOutput
  
  # 绘制 P 值条形图（按 -Log10PValue 降序排列）
  p_pvalue <- ggplot(enrichmentOutput, 
                     aes(x = reorder(Tissue, -Log10PValue),  # 按 -Log10PValue 排序
                         y = Log10PValue, 
                         label = Tissue.Specific.Genes, 
                         fill = Tissue)) +
    geom_bar(stat = "identity") +
    labs(x = "Tissue", y = "-Log10(P-Value)", title = paste("P-Value - Subtype", sub)) +
    theme_bw() +
    theme(legend.position = "none") +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.title = element_text(size = 12)) +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  # 添加到 P 值绘图列表
  pvalue_plot_list[[as.character(sub)]] <- p_pvalue
  
  # 如果有 Fold Change 数据，绘制 Fold Change 图（按 Fold.Change 降序排列）
  if (!all(is.na(enrichmentOutput$Fold.Change))) {
    p_logfc <- ggplot(enrichmentOutput, 
                      aes(x = reorder(Tissue, -Fold.Change),  # 按 Fold.Change 排序
                          y = Fold.Change, 
                          label = Tissue.Specific.Genes, 
                          fill = Tissue)) +
      geom_bar(stat = "identity") +
      labs(x = "Tissue", y = "Fold Change", title = paste("Fold Change - Subtype", sub)) +
      theme_bw() +
      theme(legend.position = "none") +
      theme(plot.title = element_text(hjust = 0.5, size = 14),
            axis.title = element_text(size = 12)) +
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank())
    
    # 添加到 Fold Change 绘图列表
    logfc_plot_list[[as.character(sub)]] <- p_logfc
  } else {
    warning(paste("No Fold Change plot generated for subtype", sub))
  }
}

# 过滤掉空的绘图对象
pvalue_plot_list <- pvalue_plot_list[!sapply(pvalue_plot_list, is.null)]
logfc_plot_list <- logfc_plot_list[!sapply(logfc_plot_list, is.null)]

# 生成组合图 - P 值
if (length(pvalue_plot_list) > 0) {
  combined_pvalue_plot <- plot_grid(plotlist = pvalue_plot_list,
                                    ncol = 2,
                                    labels = "AUTO",
                                    label_size = 12)
  
  # 保存 P 值大图
  ggsave(filename = "Tissue_Enrichment_PValue_combined_significant.png",
         plot = combined_pvalue_plot,
         width = 20,
         height = 5 * ceiling(length(pvalue_plot_list) / 2),
         dpi = 300,
         limitsize = FALSE)
}

# 生成组合图 - Fold Change
if (length(logfc_plot_list) > 0) {
  combined_logfc_plot <- plot_grid(plotlist = logfc_plot_list,
                                   ncol = 2,
                                   labels = "AUTO",
                                   label_size = 12)
  
  # 保存 Fold Change 大图
  ggsave(filename = "Tissue_Enrichment_FoldChange_combined_significant.png",
         plot = combined_logfc_plot,
         width = 20,
         height = 5 * ceiling(length(logfc_plot_list) / 2),
         dpi = 300,
         limitsize = FALSE)
} else {
  warning("No valid Fold Change plots generated - Fold Change data might be missing.")
}

# 加载所需包
library(org.Hs.eg.db)
library(clusterProfiler)
library(ggplot2)
library(dplyr)
# library(DOSE)     # 用于 dotplot
# library(cowplot)  # 用于组合多张图

# 读取数据
data <- read.csv('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/t_stats_by_subtype.csv')

# 将 SYMBOL 转换为 ENTREZID
gene_entrezid <- bitr(geneID = data$biom, 
                      fromType = "SYMBOL", 
                      toType = "ENTREZID",
                      OrgDb = "org.Hs.eg.db")

# 合并转换后的ID与原始数据
data_with_entrez <- merge(data, gene_entrezid, 
                          by.x = "biom", 
                          by.y = "SYMBOL",
                          all.x = FALSE)

# 对每个 subtype 进行 GSEA 分析并绘制 dotplot
subtypes <- unique(data_with_entrez$subtype)
gsea_results <- list()
dotplot_list <- list()

# 对每个 subtype 进行分析
for(sub in subtypes) {
  # 筛选当前 subtype 的数据
  sub_data <- data_with_entrez[data_with_entrez$subtype == sub, ]
  
  # 创建基因列表
  genelist <- sub_data$t
  names(genelist) <- sub_data$ENTREZID
  
  # 排序基因列表（从大到小）
  genelist <- sort(genelist, decreasing = TRUE)
  
  # 运行 gseGO 分析
  gsea_res <- gseGO(geneList = genelist,
                    ont = "BP",                # BP, MF, CC 全部
                    keyType = "ENTREZID",       # 输入的是 ENTREZID
                    minGSSize = 3,              # 最小基因集大小
                    maxGSSize = 3000,            # 最大基因集大小
                    pvalueCutoff = 0.05,        # p 值阈值
                    verbose = TRUE,             # 显示运行信息
                    OrgDb = org.Hs.eg.db,       # 使用人类数据库
                    pAdjustMethod = "BH")       # 使用 Benjamini-Hochberg 校正
  
  # 存储结果
  gsea_results[[sub]] <- gsea_res
  
  # 检查结果是否为空
  if(nrow(gsea_res@result) > 0) {
    # 生成 dotplot，按正负富集分面
    p <- dotplot(gsea_res, 
                 showCategory = 10,        # 显示前 10 个类别
                 split = ".sign") +        # 按正负富集分开
      facet_grid(. ~ .sign) +          # 分面显示
      ggtitle(paste("GSEA Dotplot -", sub))  # 添加标题
    
    # 添加到绘图列表
    dotplot_list[[sub]] <- p
    print(p)
  } else {
    warning(paste("No significant GO terms found for subtype:", sub))
  }
}

# 从 gsea_results 提取数据
extracted_data_list <- list()
for(sub in subtypes) {
  gsea_res <- gsea_results[[sub]]
  if(nrow(gsea_res@result) > 0) {
    result_df <- gsea_res@result
    result_df$.sign <- ifelse(result_df$NES > 0, "activated", "suppressed")
    
    # 计算 GeneRatio
    result_df$GeneRatio <- sapply(strsplit(result_df$core_enrichment, "/"), length) / result_df$setSize
    
    extracted_data <- result_df %>%
      # arrange(p.adjust) %>%
      # slice_head(n = 10) %>%
      select(Description, .sign, p.adjust, GeneRatio) %>%  # 添加 GeneRatio
      mutate(subtype = sub)
    
    extracted_data_list[[sub]] <- extracted_data
  }
}

# 合并所有 subtype 的提取数据
final_data <- bind_rows(extracted_data_list)

# 保存csv文件
write.csv(final_data, "output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/gsea_results.csv", row.names = FALSE)

# !!! 此处保存的结果表格中的 Subtype 已根据 HR 进行置换

# # 取 Description 的交集并按字母排序
# descriptions <- lapply(extracted_data_list, function(x) x$Description)
# common_descriptions <- Reduce(union, descriptions)
# common_descriptions <- sort(common_descriptions)  # 按字母排序

# # 筛选 final_data 中 Description 在交集中的数据，并计算 signed log(FDR)
# heatmap_data <- final_data %>%
#   filter(Description %in% common_descriptions) %>%
#   mutate(signed_log_fdr = ifelse(.sign == "activated", -log10(p.adjust), log10(p.adjust)))

# # 绘制热图
# p_heatmap <- ggplot(heatmap_data, aes(x = Subtype, y = Description, fill = signed_log_fdr)) +
#   geom_tile(color = "white") +
#   scale_fill_gradient2(
#     low = "#2980b9", mid = "white", high = "#c0392b", midpoint = 0,
#     limits = c(-max(abs(heatmap_data$signed_log_fdr)), max(abs(heatmap_data$signed_log_fdr))),
#     name = "Signed log10(FDR)"
#   ) +
#   # scale_x_discrete(expand = c(0, 0)) +  # 防止 x 轴拓展
#   # scale_y_discrete(expand = c(0, 0)) +  # 防止 y 轴拓展（可选）
#   theme_minimal() +
#   theme(
#     panel.grid = element_blank(),          # 删除所有 grid
#     axis.text.x = element_text(angle = 0, hjust = 1),
#     axis.text.y = element_text(size = 8)
#   ) +
#   labs(x = "Subtype", y = "Description", title = "GSEA Heatmap: Signed log10(FDR)")


# # 显示热图
# print(p_heatmap)

# ggsave(filename = "GSEA_heatmap_signed_log_fdr.png",
#        plot = p_heatmap,
#        width = 8,
#        height = 8,
#        dpi = 300,
#        limitsize = FALSE)
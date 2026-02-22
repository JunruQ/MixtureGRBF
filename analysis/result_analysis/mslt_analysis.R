library(mstate)

mslt_path = 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/multistage_life_table.csv'

data <- read_csv(mslt_path, show_col_types = FALSE)

# 定义转移矩阵
tmat <- transMat(x = list(c(2, 3), c(3), c()), names = c("Health", "Disease", "Death"))

# 转换为 mstate 数据格式
msdata <- msprep(time = "time", status = "trans", data = data, trans = tmat)

# 拟合 Aalen-Johansen 模型
cph <- coxph(Surv(Tstart, Tstop, status) ~ strata(trans), data = msdata)
msfit_obj <- msfit(cph, trans = tmat)
library(arrow)

N <- 10000
set.seed(1)
df <- data.frame(
  id = 1:N,
  TO = rlnorm(N, meanlog = log(5e6)),
  NUTS3 = sample(paste0("NUTS3_", 1:2), N, replace = TRUE),
  NACE = sample(paste0("NACE_", 1:2), N, replace = TRUE)
)

df$WAGES <- df$TO * runif(N, 0.1, 0.5)

summary(df)
write_parquet(df, "data-raw/data.parquet")

M <- 1e4

i <- sample(N, size = M, replace = TRUE)
j <- sample(N, size = M, replace = TRUE)

valid <- i != j
i <- i[valid]
j <- j[valid]

d_suplier <- df[i,]
d_user <- df[j,]

d <- data.frame(
  id_suplier = d_suplier$id,
  id_user = d_user$id,
  TO_DIFF = d_suplier$TO - d_user$TO,
  WAGES_DIFF = d_suplier$WAGES - d_user$WAGES,
  NUTS3_same = d_suplier$NUTS3 == d_user$NUTS3,
  NACE_same = d_suplier$NACE == d_user$NACE
)

d |> write_parquet("data-raw/pairs.parquet")

library(plyr)
library(dplyr)
library(stringr)
library(Hmisc)
library(ggplot2)
library(gridExtra)

capitalize_each_word <- function(word) {
  sapply(word, function(x) {
    paste(sapply(strsplit(x, ' '), capitalize), collapse=' ')
  })
}

load_data <- function(csv_file) {
  df <- read.csv(csv_file, sep="\t")
  drop_english_results(df)
}

drop_english_results <- function(df) {
  # There are only a few rows for a very small English
  # dataset that crept into these results.  Just drop
  # them.
  filter(df, dataset != "en")
}

preprocess_data <- function(df) {
  df <- preprocess_brands(df)
  df <- preprocess_dictionaries(df)
  df
}

preprocess_brands <- function(df) {
  df$dataset <- str_replace_all(df$dataset, '-', ' ')
  df$dataset <- str_replace_all(df$dataset, '(brand names) (.*)', '\\2 brands')
  df$dataset <- capitalize_each_word(df$dataset)
  df
}

preprocess_dictionaries <- function(df) {
  df$dataset <- str_replace_all(df$dataset, '^Br$', 'Breton')
  df$dataset <- str_replace_all(df$dataset, '^Ca$', 'Catalan')
  df$dataset <- str_replace_all(df$dataset, '^Cs$', 'Czech')
  df$dataset <- str_replace_all(df$dataset, '^Cy$', 'Welsh')
  df$dataset <- str_replace_all(df$dataset, '^De$', 'German')
  df$dataset <- str_replace_all(df$dataset, '^Es$', 'Spanish')
  df$dataset <- str_replace_all(df$dataset, '^Et$', 'Estonian')
  df$dataset <- str_replace_all(df$dataset, '^Fr$', 'French')
  df$dataset <- str_replace_all(df$dataset, '^Ga$', 'Irish (Gaeilge)')
  df$dataset <- str_replace_all(df$dataset, '^Hsb$', 'Upper Sorbian')
  df$dataset <- str_replace_all(df$dataset, '^Is$', 'Icelandic')
  df$dataset <- str_replace_all(df$dataset, '^It$', 'Italian')
  df$dataset <- str_replace_all(df$dataset, '^Nl$', 'Dutch')
  df$dataset <- str_replace_all(df$dataset, '^Sv$', 'Swedish')
  df
}

plot_brands <- function(df) {
  df_brands <- df[grepl("Brand|Cities", df$dataset), ]
  brands <- ggplot(subset(df_brands, model=="ConvNet"), aes(x=p1, fill=dataset))
  brands <- brands + geom_histogram(aes(y=..density..),
    position=position_dodge(width=0.075),
    binwidth=1/10,
    alpha=0.75)
  pdf("figures/brands-convnet.pdf")
  print(brands)
  dev.off()
}

plot_languages <- function(df) {
  plot_datasets <- c("German", "Italian", "Swedish", "French")
  df_plot <- filter(df, dataset %in% plot_datasets)
  plots <- list()
  df_plot$Language <- df_plot$dataset
  for (m in unique(df_plot$model)) {
    if (m == "LM") {
      mlab = "Trigram language model"
    } else {
      mlab = m
    }
    df_model <- filter(df_plot, model == m)
    p <- ggplot(data=df_model, aes(x=p1, fill=Language), alpha=0.5)
    p <- p + geom_density(alpha=0.5)
    p <- p + labs(title=mlab, x="Probability word is in English")
    p <- p + coord_cartesian(ylim=c(0, 10))
    plots[[m]] <- p
  }
  pdf("figures/languages-convnet-vs-lm.pdf")
  do.call(grid.arrange, c(plots, list(ncol=1)))
  dev.off()
}

run <- function(csv_file="models/artificial_errors/error_analysis.csv") {
  df <- load_data(csv_file)
  df <- preprocess_data(df)
  plot_brands(df)
  plot_languages(df)
  df
}

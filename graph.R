library(tidyr)
library(readr)
library(ggplot2)
library(dplyr)
library(stringr)

df <- read.csv("out.csv")

df %>% ggplot(aes(step)) + geom_point(aes(x=step, y=value, color=monitor))

df %>% filter(str_detect(monitor, "not normalized") | str_detect(monitor, "cosine")) %>% ggplot(aes(step)) + geom_point(aes(x=step, y=value, color=monitor))
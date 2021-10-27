# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 2021/10/24


install.packages("igraph")

install.packages("tidyverse")

# Library
library(igraph)
library(tidyverse)
options(scipen=200)


set.seed(1)

data <- read_csv('anti_data.csv')  #全量数据

data <- data %>% mutate(relat =case_when(phase==10 ~ '好吃',
                                         phase==20 ~ '不好吃'))

relations <- data.frame(from=data$uid,
                        to=data$did,
                        status=data$relat)

network <- graph_from_data_frame(relations, directed = TRUE, vertices = NULL )
print(network, e=TRUE, v=FALSE)

# Default network
par(mar=c(0,0,0,0))

library(parallel)
detectCores()

#E(network)$size <- 0.5
# 社区检测
ceb <- cluster_edge_betweenness(network) # 构造模型
dendPlot(ceb) # 可视化

# 将组别存储
group = data.frame(uid=0,group=0)
for (i in (1:length(ceb))) {
        tmp = data.frame(uid=ceb[i],group=i)
        colnames(tmp) <- c('uid','group')
        colnames(group) <- c('uid', 'group')
        group <- rbind2(group,tmp)
}
group <- group[-1,]
# write_excel_csv(group,'group.csv')

write_excel_csv(group,'test.csv')

png(file="graph.png",
    width=8000, height=8000)
plot(ceb,network,
     layout=layout.auto,
     vertex.shape=c("circle","square"),             # One of “none”, “circle”, “square”, “csquare”, “rectangle” “crectangle”, “vrectangle”, “pie”, “raster”, or “sphere”
     vertex.size=1.5,                          # Size of the node (default is 15)
     vertex.size2=0.5,                               # The second size of the node (e.g. for a rectangle)
     vertex.label = NA,
     edge.arrow.size=0.1,
     edge.arrow.width = 0.1
     #label = NA
     )

dev.off()

plot(ceb,network,
     layout=layout.auto,
     vertex.shape=c("circle","square"),             # One of “none”, “circle”, “square”, “csquare”, “rectangle” “crectangle”, “vrectangle”, “pie”, “raster”, or “sphere”
     vertex.size=1.5,                          # Size of the node (default is 15)
     vertex.size2=0.5,                               # The second size of the node (e.g. for a rectangle)
     vertex.label = NA,
     edge.arrow.size=0.1,
     edge.arrow.width = 0.1
     #label = NA
     )

# 读取保存uid
group <- read_csv("grouph.csv")

group

# 输出uid
group_s <- dplyr::filter(group,group==2)




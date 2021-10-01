dat = read.table("../results/model-accuracy.csv", header=TRUE, sep=",")
##str(dat)

data = rep("S.A.500",length(dat$modelfile))
model = rep("NN",length(dat$modelfile))
for(i in 1:length(dat$modelfile)){
  if(dat$modelnum[i] == 4)
      model[i] = "logistic"
  if(dat$modelfile[i] == "model-staph.jl" & dat$nvar[i] == 700){
    data[i] = "S.A.700"
  }else if(dat$modelfile[i] == "model-pseudomonas-sequences.jl" & dat$nvar[i] == 500){
    data[i] = "P.A.500"
  }else if(dat$modelfile[i] == "model-pseudomonas-sequences.jl" & dat$nvar[i] == 700){
    data[i] = "P.A.700"
  }else if(dat$modelfile[i] == "model-pseudomonas-images.jl"){
    data[i] = "P.A.Images"
  }else if(dat$modelfile[i] == "model-staph-randForest.jl" & dat$nvar[i] == 700){
    data[i] = "S.A.700"
    model[i] = "RF"
  }else if(dat$modelfile[i] == "model-staph-randForest.jl" & dat$nvar[i] == 500){
    model[i] = "RF"
  }else if(dat$modelfile[i] == "model-pseudo-seq-randForest.jl" & dat$nvar[i] == 500){
    data[i] = "P.A.500"
    model[i] = "RF"
  }else if(dat$modelfile[i] == "model-pseudo-seq-randForest.jl" & dat$nvar[i] == 700){
    data[i] = "P.A.700"
    model[i] = "RF"
  }else if(dat$modelfile[i] == "model-pseudo-images-randForest.jl"){
    data[i] = "P.A.Images"
    model[i] = "RF"
  }
}

dat$data = data
dat$model = paste0(model,dat$modelnum-1)
dat$model[dat$data == "P.A.Images" & dat$model == "NN0"] = "NN1"


dat3 = subset(dat, model != "NN0")
dat2 = subset(dat3, data != "P.A.Images")
ind1 = which(dat2$model == "logistic3")
ind2 = which(dat2$model == "RF0")
ind3 = setdiff(1:length(dat2$model),c(ind1,ind2))
mlog = mean(dat2$accuracy[ind1])
mrf = mean(dat2$accuracy[ind2])
mnn = mean(dat2$accuracy[ind3])

#dat2 = within(dat2, model<-factor(model, labels=c("logistic", "unregularized-NN1", "unregularized-NN2", "regularized-NN1", "regularized-NN2", "RF")))

dat2 = within(dat2, model<-factor(model, labels=c("logistic", "unregularized-NN1", "unregularized-NN2", "regularized-NN1", "regularized-NN2","RF")))
dat2 = subset(dat2, model!="RF")
dat2 = within(dat2, model<-factor(model, labels=c("logistic", "unregularized-NN1", "unregularized-NN2", "regularized-NN1", "regularized-NN2")))

p = ggplot(dat2,aes(x=data,y=accuracy,size=time,color=model))+geom_jitter(width=0.2)+ylim(0.4,1.0)+xlab("")+
  scale_size_continuous(guide=FALSE)+
  geom_hline(yintercept=0.5,linetype=3)+
  theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="grey", size=rel(1.5), angle=90, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="grey", size=rel(1.5), angle=90, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.0)), legend.title=element_text(size=rel(1.0)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        legend.position="top"
        )

q = ggplot(dat2,aes(x=data,y=time,color=model))+geom_point()+xlab("")+
  theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(2.1)),
        axis.title.y = element_text(size=rel(2.1), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="grey", size=rel(1.8), angle=90, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="grey", size=rel(1.8), angle=90, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.5)), legend.title=element_text(size=rel(1.5)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        legend.position="top"
        )

pdf("Figures/accuracy.pdf", width=9, height=5)
p
dev.off()
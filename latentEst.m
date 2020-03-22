function latentX = latentEst(barK,barC,barR,bary,bard)

latentX = barK * barC' / (barC * barK * barC' + barR) * (bary - bard);

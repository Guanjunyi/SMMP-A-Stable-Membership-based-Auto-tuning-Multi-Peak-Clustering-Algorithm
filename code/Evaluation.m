function [AMI,ARI,FMI] = Evaluation(cl,answer)
import java.util.LinkedList
import Library.*
if~isempty(answer)
        AMI=GetAmi(answer,cl);
        ARI=GetAri(answer,cl);
        FMI=GetFmi(answer,cl);
else
    AMI=nan;
    ARI=nan;
    FMI=nan;
end
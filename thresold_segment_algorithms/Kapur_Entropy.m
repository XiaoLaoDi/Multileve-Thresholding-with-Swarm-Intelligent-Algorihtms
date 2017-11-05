function result=Kapur_Entropy(LP,TH)
%function result=Kapur_Entropy(LP,TH)
    %calculate the maximum entropy based on the input thresholds
%input:
    %LP:normalized histogram
    %TH:thresholds
%output:
    %result:the entropy
    clear k i j
    k=size(TH,2);
    result=0;
    s_th=[0 TH length(LP)];
    s=sort(s_th);
    for i=1:k+1
        n1=s(i)+1;
        n2=s(i+1);
        sub_sum=0;
        sum_sub=sum(LP(n1:n2));
        for j=n1:n2
            if LP(j)==0
                continue;
            end
            sub_sum=sub_sum-(LP(j)./sum_sub)*log(LP(j)./sum_sub);
        end
        result=result+sub_sum;
    end
end
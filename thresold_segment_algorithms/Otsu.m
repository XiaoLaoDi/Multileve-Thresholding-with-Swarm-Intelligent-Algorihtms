function result=Otsu(LP,TH)

%function result=Otsu(LP,TH)
    %calculate the maximum between-variance based on the input thresholds
%input:
    %LP:normalized histogram
    %TH:thresholds
%output:
    %result:the Otsu-based maximum inter-class variance
    
%%程序的正确性已由“Cuckoo Search and Firefly Algorithm Applied to Multilevel Image
%%Thresholding”中的"livingroom.tif"图片验证
    
GrayLevel=0:length(LP)-1;
mean_gray=sum(GrayLevel.*LP);
    
Thresholds=sort(TH);
Thresh_ex=[0 Thresholds length(LP)];

sum_sum=zeros(1,length(TH)+1);
for i=1:length(TH)+1
    sub_sum=sum(LP(Thresh_ex(i)+1:Thresh_ex(i+1)));
    if sub_sum==0
        sum_sum(i)=0;
        continue;
    end
    sub_gray=Thresh_ex(i):Thresh_ex(i+1)-1;
    sum_sub=sum(sub_gray.*(LP(Thresh_ex(i)+1:Thresh_ex(i+1))./sub_sum));
    sum_sum(i)=sub_sum * (sum_sub - mean_gray) * (sum_sub - mean_gray);
end
sigma_local=sum(sum_sum);

result=sigma_local;

end
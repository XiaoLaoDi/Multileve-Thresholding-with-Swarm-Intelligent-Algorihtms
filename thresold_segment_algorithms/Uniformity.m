function U = Uniformity(Raw_img,Thresholds)

% calculation reference paper---" A fast scheme for optimal thresholding using genetic algorithms "
L=256;
TH_NUM=length(Thresholds);

[numb,pixle]=imhist(Raw_img,L);
[a,b]=size(Raw_img);
ImgSize=a*b;
G_max=double(max(max(Raw_img)));
G_min=double(min(min(Raw_img)));

LP=numb'/(a*b);

pixleTran=pixle';
T=[0 Thresholds L];

result=0;
for i=1:TH_NUM+1
    i_gray_level=pixleTran(T(i)+1:T(i+1));
    sub_LP=LP(T(i)+1:T(i+1))./sum(LP(T(i)+1:T(i+1)));
    class_i_mean_grayvalue=sum(i_gray_level.*sub_LP);
    sub_sum=(i_gray_level-class_i_mean_grayvalue).*(i_gray_level-class_i_mean_grayvalue);
    sub_result=sum(sub_sum);
    result=result+sub_result;
end

% U=1-2*TH_NUM*result/(ImgSize*(G_max-G_min).^2);%ԭ�Ĳ�������ͼ��ϴ�ʱ������ֱ�۶Ա���,����TH_NUMӰ��
U=1-2*result/(ImgSize*(G_max-G_min).^2);%Uniformity�������ԭ��

%ִ�����ɱ任͹�Ա仯
Const=1;
EXP=log10(ImgSize/10000)*10.^4;
U=Const*U.^EXP;

end
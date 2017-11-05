function [X_new,Fitness_new,PS_SUCCESS_FLAG,Count]=MyPatternSearch(CalculateFitnessFunction,X_old,fitness_old,TH_num,Count)
% function [X_new,Fitness_new,PS_SUCCESS_FLAG,Count]=MyPatternSearch(CalculateFitnessFunction,X_old,fitness_old,TH_num,Count)
% Input:
%       X_old:��ʼ��
%       fitness_old:��ʼ����Ӧ��ֵ
%       TH_num�����ά��/��ֵ��
% Output:
%       X_new:ģʽ������ĵ�
%       Fitness_new:ģʽ����������Ӧ��ֵ

global nd st LP;
fitness=CalculateFitnessFunction;
% �㷨������
         delta=1;   %��=1   ��ʼ���� 
         alpha=1;   %��=1   ��������
%          beta=1/2;  %��=1/2 ������
%          eps=1;     %�������         
         PS_SUCCESS_FLAG=0;
x0=X_old;
Fitness0=fitness_old;

while(1)
%     if delta<eps
%         break;
%     end
    [x1,Fitness1,Count]=DetectMove(fitness,x0,Fitness0,delta,TH_num,Count); %�㷨ִ�С�̽���ƶ����ӡ�
    if x1==x0 %̽������̽��ʧ��
%         delta=beta*delta;
        break;
    else %ִ�С�ģʽ�ƶ����ӡ�
        while(1)
            xp=x1+alpha.*(x1-x0);
                xp(xp>nd)=nd;
                xp(xp<st)=st;
            FitnessXp=1/fitness(LP,xp);
            Count=Count+1;
            if FitnessXp<Fitness1          
                x0=x1;
                Fitness0=Fitness1;
                x1=xp;
                Fitness1=FitnessXp;
                Count=Count+1;
            else
                x0=x1;
                Fitness0=Fitness1;
                break;
            end %end if FitnessXp<Fitness1
        end %end while(1)
    end %end if x1==x0
end % end while(1)

X_new=x0;
Fitness_new=Fitness0;
if Fitness_new<fitness_old
    PS_SUCCESS_FLAG=1;
end

end

function[X_Detect,FitnessDetect,Count]=DetectMove(CalculateFitnessFunction,x,Fitness,alpha,TH_num,Count)
global nd st LP;
fitness=CalculateFitnessFunction;
E=eye(TH_num);%������������
x1=x;
Fitness1=Fitness;
for i=1:TH_num   
    x2=x1+alpha.*E(i,:);
    x2(x2>nd)=nd;
    x2(x2<st)=st;
    X2Fitness=1/fitness(LP,x2);
    if X2Fitness<Fitness1
        x1=x2;
        Fitness1=X2Fitness; 
    else
        x2=x1-alpha.*E(i,:);
        x2(x2>nd)=nd;
        x2(x2<st)=st;
        X2Fitness=1/fitness(LP,x2);
        if X2Fitness<Fitness1
            x1=x2;
            Fitness1=X2Fitness;
        end
        Count=Count+1;
    end
    Count=Count+1;
end

X_Detect=x1;
FitnessDetect=Fitness1;

end
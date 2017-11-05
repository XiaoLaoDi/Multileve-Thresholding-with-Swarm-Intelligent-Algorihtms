function [X_new,Fitness_new,PS_SUCCESS_FLAG,Count]=MyPatternSearch(CalculateFitnessFunction,X_old,fitness_old,TH_num,Count)
% function [X_new,Fitness_new,PS_SUCCESS_FLAG,Count]=MyPatternSearch(CalculateFitnessFunction,X_old,fitness_old,TH_num,Count)
% Input:
%       X_old:初始点
%       fitness_old:初始点适应度值
%       TH_num：解的维数/阈值数
% Output:
%       X_new:模式搜索后的点
%       Fitness_new:模式搜索后点的适应度值

global nd st LP;
fitness=CalculateFitnessFunction;
% 算法参数：
         delta=1;   %δ=1   初始步长 
         alpha=1;   %α=1   加速因子
%          beta=1/2;  %β=1/2 缩减率
%          eps=1;     %允许误差         
         PS_SUCCESS_FLAG=0;
x0=X_old;
Fitness0=fitness_old;

while(1)
%     if delta<eps
%         break;
%     end
    [x1,Fitness1,Count]=DetectMove(fitness,x0,Fitness0,delta,TH_num,Count); %算法执行“探测移动算子”
    if x1==x0 %探测算子探测失败
%         delta=beta*delta;
        break;
    else %执行“模式移动算子”
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
E=eye(TH_num);%基本轴向搜索
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
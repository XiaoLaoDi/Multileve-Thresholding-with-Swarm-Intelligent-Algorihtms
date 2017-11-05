function  PSO_result=PSO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function PSO_result=PSO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The Particle Swarm Optimization,proposed by J.Kennedy and Russ C.Eberhart in 1995,is try to optimize some minimization problems
%    Calculating the maximum Kapur_Entropy/Between-Variance use the Particle Swarm Opimization Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    PSO_result: is a 'Struct' containing the following result information
%        PSO_result.Fitness:
%            PSO_result.Fitness.Mean
%            PSO_result.Fitness.Variance
%            PSO_result.Fitness.Max
%            PSO_result.Fitness.Min
%        PSO_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        PSO_result.BestThresholds: a vector containing 'TH_num' values
%        PSO_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        PSO_result.MeanConvergenceTime: the mean convergence time

disp('the Particle Swarm Optimization is running...')

%% PSO STEPS
% STEP ONE
% Initialize the PSO parameters
global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
fitness=FunctionName;
T=MAX_Iterations;
D=TH_num;
N=40;
v_max=5;                                          % v_max:0.1*x_max~0.2*x_max
w=0.5;

%% 预分配内存
%记录结果
PSO_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
PSO_result.EachRunBestFitness=zeros(1,RunTimes);
PSO_result.BestThresholds=zeros(1,TH_num);
PSO_result.EachRunConvergenceTime=zeros(1,RunTimes);
PSO_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
PSO_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
PSO_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让PSO跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    % STEP TWO
    %------Initialize the PSO swarm's position and velocity------------
    x=zeros(N,D);
    for i=1:N
        for j=1:D
            x(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));           %randomized position                               
        end
    end
    v=-v_max+2*v_max.*rand(N,D);                                        %randomized velocity

    %------calculate fitness ----------------------
    p=zeros(1,N);
    y=zeros(N,D);
    for i=1:N
        p(i)=1/fitness(LP,x(i,:));
        FunCount=FunCount+1;
        y(i,:)=x(i,:);
    end
    % STEP THREE
    %------the PSO main iterations ------------
    Each_Iterate_BestFitness=zeros(1,T);
    Each_Iterate_BestThresholds=zeros(T,D);
    Iterates=1;
    Success_FindBest_Index(RepeatTimes)=0;
    while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)        
        % iterations
        [Each_Iterate_BestFitness(Iterates),min_index]=min(p);
        Each_Iterate_BestThresholds(Iterates,:)=x(min_index,:);
        if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
            PSO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
            EachRunFunCalNum(Iterates)=FunCount;
            Success_FindBest_Index(RepeatTimes)=1;
            break;
        end   
        for i=1:N
            v(i,:)=round(w*v(i,:)+2.*rand(1,D).*(y(i,:)-x(i,:))+2.*rand(1,D).*(Each_Iterate_BestThresholds(Iterates,:)-x(i,:)));             
            for j=1:D
                if abs(v(i,j))>v_max
                    v(i,j)=round(sign(v(i,j))*v_max);
                end
            end
            x(i,:)=x(i,:)+v(i,:);
            % boundary detect
                x(x>nd)=nd;
                x(x<st)=st;
            UpdatedPi=1/fitness(LP,x(i,:));
            FunCount=FunCount+1;
            if  UpdatedPi < p(i)
                p(i)=UpdatedPi;
                y(i,:)=x(i,:);
                if p(i) < Each_Iterate_BestFitness(Iterates)
                    Each_Iterate_BestThresholds(Iterates,:)=y(i,:);
                    Each_Iterate_BestFitness(Iterates)=p(i);
                end
            end                     
        end
        PSO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Iterates=Iterates+1;
    end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"   
    
%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    PSO_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    PSO_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    PSO_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    PSO_result.EachRunConvergenceTime(RepeatTimes)=PSO_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    PSO_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [PSO_result.Fitness.Max,PSOFMax_Index]=max(PSO_result.EachRunBestFitness);
    [PSO_result.Fitness.Min,~]=min(PSO_result.EachRunBestFitness);
    PSO_result.Fitness.Mean=mean(PSO_result.EachRunBestFitness);
    PSO_result.Fitness.Variance=var(PSO_result.EachRunBestFitness);
    PSO_result.BestThresholds=sort(PSO_result.EachRunBestThresholds(PSOFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    PSO_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    PSO_result.MeanConvergenceTime=mean(PSO_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存PSO结果：ImageNmae_THChar_PSO_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_PSO_result.mat');
    save(FILENAME,'PSO_result');     

disp('the Particle Swarm Optimization is accomplised !!!')



end
function  ABC_result=ABC_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function ABC_result=ABC_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    D. Karaboga, AN IDEA BASED ON HONEY BEE SWARM FOR NUMERICAL OPTIMIZATION,TECHNICAL REPORT-TR06, 
%    Erciyes University, Engineering Faculty, Computer Engineering Department 2005.
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    ABC_result: is a 'Struct' containing the following result information
%        ABC_result.Fitness:
%            ABC_result.Fitness.Mean
%            ABC_result.Fitness.Variance
%            ABC_result.Fitness.Max
%            ABC_result.Fitness.Min
%        ABC_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        ABC_result.BestThresholds: a vector containing 'TH_num' values
%        ABC_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        ABC_result.MeanConvergenceTime: the mean convergence time

disp('the Artificial Bee Colony Algorithm is running...')

%% ABC STEPS
% STEP ONE
% Initialize the ABC parameters
global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
fitness=FunctionName;
T=MAX_Iterations;
D=TH_num;
N=20;
limit=50;

%% 预分配内存
ABC_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
ABC_result.EachRunBestFitness=zeros(1,RunTimes);
ABC_result.BestThresholds=zeros(1,TH_num);
ABC_result.EachRunConvergenceTime=zeros(1,RunTimes);
ABC_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
ABC_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
ABC_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);

%% 最外层循环，让ABC跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T); 
    
    % STEP TWO
    %------Initialize the ABC swarm's position and velocity------------
    x=zeros(N,D);
    trial=zeros(1,N);
    for i=1:N
        for j=1:D
            x(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));             % randomized position 
        end
    end
    
    p=zeros(1,N);
    % calculate the initialized fitness
        for i=1:N
            p(i)=1/fitness(LP,x(i,:));
            FunCount=FunCount+1;
        end   
        
    % STEP THREE
    %------the ABC main iterations ------------
    Each_Iterate_BestFitness=zeros(1,T);
    Each_Iterate_BestThresholds=zeros(T,D);
    Success_FindBest_Index(RepeatTimes)=0;
    Iterates=1;

    while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)               
        % record the best
            [Each_Iterate_BestFitness(Iterates),min_index]=min(p);
            Each_Iterate_BestThresholds(Iterates,:)=x(min_index,:);
            Each_Iterate_BestThresholds(Iterates,:)=Each_Iterate_BestThresholds(Iterates,:);
            if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                ABC_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                EachRunFunCalNum(Iterates)=FunCount;
                Success_FindBest_Index(RepeatTimes)=1;
                break;
            end         
       %% Employed Bees' Phase
            x_new=zeros(N,D);
            for i=1:N
                j=randi([1 D]);
                k=randi([1 N]);
                while(k==i)
                    k=randi([1 N]);
                end
%                 Phi=2*rand-1;
                Phi=0.4*(rand-0.5);
                x_new(i,j)=x(i,j)+Phi*(x(i,j)-x(k,j));
                x_new(i,:)=round(x_new(i,:));
                % boundary detect
                    for j=1:D
                        if x_new(i,j) > nd
                            x_new(i,j)=nd;
                        end
                        if x_new(i,j) < st
                            x_new(i,j)=st;
                        end
                    end
                UpdatedPi=1/fitness(LP,x_new(i,:));
                FunCount=FunCount+1;
                if  UpdatedPi < p(i)
                    p(i)=UpdatedPi;
                    x(i,:)=x_new(i,:);
                    trial(i)=0;
                else
                    trial(i)=trial(i)+1;
                end
            end
        %% Onlooker Bees' Phase
%             ReMappedFitness=FitnessMapFunc2ABC(p);
%             Ps=ReMappedFitness./sum(ReMappedFitness);                   % Calculate probabilities for onlookers by roulette-wheel-like selection
            Ps=p./sum(p);
            s=1;
            t=0;
            while(t<N)
                r=rand;
                if r<Ps(s)
                    t=t+1;
                    j=randi([1 D]);
                    k=randi([1 N]);
                    while(k==s)
                        k=randi([1 N]);
                    end
%                     Phi=2*rand-1;
                    Phi=0.4*(rand-0.5);
                    x_new(s,j)=x(s,j)+Phi*(x(s,j)-x(k,j));
                    x_new(s,:)=round(x_new(s,:));
                    % boudary detect
                        for j=1:D
                            if x_new(s,j) > nd
                                x_new(s,j)=nd;
                            end
                            if x_new(s,j) < st
                                x_new(s,j)=st;
                            end
                        end
                    UpdatedPi=1/fitness(LP,x_new(s,:));
                    FunCount=FunCount+1;
                    if  UpdatedPi < p(s)
                        p(s)=UpdatedPi;
                        x(s,:)=x_new(s,:);
                        trial(s)=0;
                    else
                        trial(s)=trial(s)+1;
                    end
                end
                s=s+1;
                if(s==(N+1))
                    s=1;
                end
            end
        %% Scout bee phase
            [~,mi]=max(trial);
            if trial(mi)>limit
                for j=1:D
                    x(mi,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));
                end
                % boundary detect
                    for j=1:D
                        if x(mi,j) > nd
                            x(mi,j)=nd;
                        end
                        if x(mi,j) < st
                            x(mi,j)=st;
                        end
                    end
                p(mi)=1/fitness(LP,x(mi,:));
                FunCount=FunCount+1;
                trial(mi)=0;
            end
    ABC_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
    EachRunFunCalNum(Iterates)=FunCount;  
    Iterates=Iterates+1;
    end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"
%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        % 去除零元素
    ABC_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    ABC_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    ABC_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    ABC_result.EachRunConvergenceTime(RepeatTimes)=ABC_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    ABC_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [ABC_result.Fitness.Max,ABCFMax_Index]=max(ABC_result.EachRunBestFitness);
    [ABC_result.Fitness.Min,~]=min(ABC_result.EachRunBestFitness);
    ABC_result.Fitness.Mean=mean(ABC_result.EachRunBestFitness);
    ABC_result.Fitness.Variance=var(ABC_result.EachRunBestFitness);
    ABC_result.BestThresholds=sort(ABC_result.EachRunBestThresholds(ABCFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    ABC_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    ABC_result.MeanConvergenceTime=mean(ABC_result.EachRunConvergenceTime);


disp('Statistic Metrics is Calculated !')

% 保存ABC结果：ImageNmae_THChar_ABC_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_ABC_result.mat');
    save(FILENAME,'ABC_result');     

disp('the Artificial Bee Colony Algorithm is accomplised !!!')



end
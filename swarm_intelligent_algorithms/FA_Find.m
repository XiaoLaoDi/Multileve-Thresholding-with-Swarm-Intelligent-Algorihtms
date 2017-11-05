function  FA_result=FA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function FA_result=FA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    Firefly Algorithm (FA) developed by Xin-she Yang at Cambridge University in 2008 in the Chapter 8 of 'monograph ONE'
%    Yang,X.S.:Nature-inspired metaheuristic algorithms,Luniver Press,(2008)--refered as 'Monograph ONE'
%    Szymon Lukasik and Slawomir Zak in "Firefly Algorithm for Continuous Constrained Optimization Tasks"---refered as ‘Paper TWO’
%    Note: the MFA is to minimize some optimization problem
%    Calculating the maximum Kapur-Entropy/Otsu value use the Firefly Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    FA_result: is a 'Struct' containing the following result information
%        FA_result.Fitness:
%            FA_result.Fitness.Mean
%            FA_result.Fitness.Variance
%            FA_result.Fitness.Max
%            FA_result.Fitness.Min
%        FA_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        FA_result.BestThresholds: a vector containing 'TH_num' values
%        FA_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        FA_result.MeanConvergenceTime: the mean convergence time

disp('Firefly Algorithm is running...')

%% Pre-processing: Initializing
% STEP ONE
    % Initialize the problem parameters
        global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
        fitness=FunctionName;                            % fitness: the problem which required to maximize 
        D=TH_num;                                           % D: the Dimension of the problem
        x_max=nd;                                           % x_max: the up-boundary of each Dimension
        x_min=st;                                           % x_min: the down-limit of each Dimension 
        scale=x_max-x_min;
% STEP TWO
    % Initialize FA parameters
    % the initialized FA parameters suggested by 'Paper TWO'
        alpha=0.006;                                         %α∈[0,1]
        betamax=0.8;                                        % β0 is the maximum attractiveness value
        gamma=1;
        Num=40;                                             % Num: number of fireflies
        T=MAX_Iterations;                                   % T: maximum iterations

%% 预分配内存
FA_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
FA_result.EachRunBestFitness=zeros(1,RunTimes);
FA_result.BestThresholds=zeros(1,TH_num);
FA_result.EachRunConvergenceTime=zeros(1,RunTimes);
FA_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
FA_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
FA_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让FA跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister'); 
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    
% STEP THREE
    % Initialize the Firefly 
        x=zeros(Num,D); 
        for i=1:Num
            for j=1:D
                x(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));       % randomized position
            end
        end 
    % Calculate the absolute light intensity of the firefly swarm   
         ABS_brightness=zeros(1,Num);
         
%% Main iterations of firefly algorithm
% Main loops
    Each_Iterate_BestFitness=zeros(1,T);                    % the optimal fitness of each iteration
    Each_Iterate_BestThresholds=zeros(T,D);                 % the optimal solution of each iteration
    Iterates=1;
    Success_FindBest_Index(RepeatTimes)=0;
     while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)   
        % Find the best solution index
            for i=1:Num
                ABS_brightness(i)=1/fitness(LP,x(i,:));
                FunCount=FunCount+1;
            end
            [ABS_brightnessSorted,index]=sort(ABS_brightness);
            ABS_brightness=ABS_brightnessSorted;
            x=x(index,:);
            Each_Iterate_BestFitness(Iterates)=ABS_brightness(1);
            Each_Iterate_BestThresholds(Iterates,:)=x(1,:);
            if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                FA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                EachRunFunCalNum(Iterates)=FunCount;
                Success_FindBest_Index(RepeatTimes)=1;
                break;
            end           
        % Inner loops of FA
        for i=1:Num
            for j=1:i
                if ABS_brightness(j) < ABS_brightness(i)
                % Calculate the distance between Firefly i and Firefly j
                    r=sqrt(sum((x(i,:)-x(j,:)).^2));
                % Calculate the attractiveness of Firefly i to Firefly j
                    beta=betamax*exp(-gamma*r.^2)+(1-betamax);
                % Firefly i move to Firefly j
                    % MFA(Modified Firefly Algorithm) iterative equatio
                    x(i,:)=x(i,:)+beta*(x(j,:)-x(i,:))+alpha*scale.*(rand(1,D)-0.5); 
                    x(i,:)=round(x(i,:));
                    x(x>nd)=nd;
                    x(x<st)=st;
                end
            end
        end
         FA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
         EachRunFunCalNum(Iterates)=FunCount;
         Iterates=Iterates+1;
     end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"

%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    FA_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    FA_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    FA_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    FA_result.EachRunConvergenceTime(RepeatTimes)=FA_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    FA_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);

end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [FA_result.Fitness.Max,FAFMax_Index]=max(FA_result.EachRunBestFitness);
    [FA_result.Fitness.Min,~]=min(FA_result.EachRunBestFitness);
    FA_result.Fitness.Mean=mean(FA_result.EachRunBestFitness);
    FA_result.Fitness.Variance=var(FA_result.EachRunBestFitness);
    FA_result.BestThresholds=sort(FA_result.EachRunBestThresholds(FAFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    FA_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    FA_result.MeanConvergenceTime=mean(FA_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存FA结果：ImageNmae_THChar_FA_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_FA_result.mat');
    save(FILENAME,'FA_result');      

disp('the Firefly Algorithm is accomplised !!!')
end

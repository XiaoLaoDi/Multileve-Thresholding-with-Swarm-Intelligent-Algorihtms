function  PatternTypeTwo_BA_result=PatternTypeTwo_BA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function PatternTypeTwo_BA_result=PatternTypeTwo_BA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The PatternTypeTwo_BA,proposed by Xin-She Yang in 2010,is try to minimize some optimization problems
%    Calculating the maximum Kapur-Entropy/Otsu value use the Firefly Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    PatternTypeTwo_BA_result: is a 'Struct' containing the following result information
%        PatternTypeTwo_BA_result.Fitness:
%            PatternTypeTwo_BA_result.Fitness.Mean
%            PatternTypeTwo_BA_result.Fitness.Variance
%            PatternTypeTwo_BA_result.Fitness.Max
%            PatternTypeTwo_BA_result.Fitness.Min
%        PatternTypeTwo_BA_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        PatternTypeTwo_BA_result.BestThresholds: a vector containing 'TH_num' values
%        PatternTypeTwo_BA_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        PatternTypeTwo_BA_result.MeanConvergenceTime: the mean convergence time
disp('the Pattern Type Two Bat Algorithm is running...')

%% Bat Algorithm STEPS
    %% STEP ONE
        % Initialize the PatternTypeTwo_BA parameters
            % The following parameters in algorithm recommended by monograph
            % " Yang,X.S.:Nature-inspired metaheuristic algorithms,Luniver Press,(2008)" Cited as Monograph One
            global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
            fitness=FunctionName;
            N=40;                                                   % Number of bats (or different solutions)
            freq_max=2;                                             % Frequency maximum
            freq_min=0;                                             % Frequency minimum
            r_0=0.5;                                                % Pulse rate
            A_0=0.5;                                                % Loudness    
            T=MAX_Iterations;                                       % Maximum iterations
            D=TH_num;                                               % the number of thresholds

PS_SUCCESS_FLAG=0;                        
            
%% 预分配内存
PatternTypeTwo_BA_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
PatternTypeTwo_BA_result.EachRunBestFitness=zeros(1,RunTimes);
PatternTypeTwo_BA_result.BestThresholds=zeros(1,TH_num);
PatternTypeTwo_BA_result.EachRunConvergenceTime=zeros(1,RunTimes);
PatternTypeTwo_BA_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
PatternTypeTwo_BA_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
PatternTypeTwo_BA_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让PatternTypeTwo_BA跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');    
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    
    %% STEP TWO
        % Random initialize the Bat Algorithm solutions
            x=zeros(N,D);
            v=zeros(N,D);
            for i=1:N
                for j=1:D
                    x(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));       % randomized position
                end
            end
        % calculate fitness, get the current best solution(i.e. bestbat) & batfitness(i.e.bestfitness)
            batfitness=zeros(1,N);
            for i=1:N
                batfitness(i)=1/fitness(LP,x(i,:));
                FunCount=FunCount+1;
            end
    %% STEP THREE
        %% the PatternTypeTwo_BA main iterations
            x_new=zeros(N,D);
            Each_Iterate_BestFitness=zeros(1,T);                    % the optimal fitness of each iteration
            Each_Iterate_BestThresholds=zeros(T,D);                 % the optimal solution of each iteration
            Iterates=1;
            Success_FindBest_Index(RepeatTimes)=0;
            while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)               
              %% STEP FOUR
                    [Each_Iterate_BestFitness(Iterates),index]=min(batfitness);
                    Each_Iterate_BestThresholds(Iterates,:)=x(min(index),:);
                    if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                        PatternTypeTwo_BA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                        EachRunFunCalNum(Iterates)=FunCount;
                        Success_FindBest_Index(RepeatTimes)=1;
                        break;
                    end     

                    
                    
X_old=Each_Iterate_BestThresholds(Iterates,:);
fitness_old=Each_Iterate_BestFitness(Iterates);
[X_new,Fitness_new,PS_SUCCESS_FLAG,FunCount]=MyPatternSearch(fitness,X_old,fitness_old,D,FunCount);
 if PS_SUCCESS_FLAG==1
     Each_Iterate_BestThresholds(Iterates,:)=X_new;
        Each_Iterate_BestFitness(Iterates)=Fitness_new;
     if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
        PatternTypeTwo_BA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Success_FindBest_Index(RepeatTimes)=1;
        break;
     end
 end                    
                    
                for i=1:N
                    % modify frequency
                        freq=freq_min+(freq_max-freq_min).*randn(1,D);
                    % update velocity
                        v(i,:)=v(i,:)+(x(i,:)-Each_Iterate_BestThresholds(Iterates,:)).*freq;
                    % update bat position
                        x_new(i,:)=x(i,:)+v(i,:);
                        x_new(i,:)=abs(round(x_new(i,:)));
                    % boundary detect
                        x_new(x_new>nd)=nd;
                        x_new(x_new<st)=st;
                    % local search
                        if rand > r_0
                            % The following parameters in algorithm recommended by paper
                            % "Adis Alihodzic, Milan Tuba.'Improved Bat Algorithm Applied to Multilevel Image Thresholding,(2014)'"cited as Paper One
                            Tur=1.66;       % taken from paper One
                                            % Tur:controls local search turbulence variance,experiments 
                                            % examined that Tur∈(1,0.1*nd) can reach satisfied results
                            x_new(i,:)=Each_Iterate_BestThresholds(Iterates,:)+(-1+2*rand(1,D)).*Tur;          % when 'Tur' larger is benificial to More-thresholds             
                            x_new(i,:)=abs(round(x_new(i,:)));
                            % boundary detect
                                x_new(x_new>nd)=nd;
                                x_new(x_new<st)=st;
                        end
                    % evaluate new bat's position and update
                        Batnew=1/fitness(LP,x_new(i,:));
                        FunCount=FunCount+1;
                        if (Batnew <= batfitness(i)) && (rand < A_0)
                            x(i,:)=x_new(i,:);
                            batfitness(i)=Batnew;
                        end
                    % update the current best bat
                        if Batnew < Each_Iterate_BestFitness(Iterates)
                            Each_Iterate_BestThresholds(Iterates,:)=x_new(i,:);
                            Each_Iterate_BestFitness(Iterates)=Batnew;
                        end
                end
                PatternTypeTwo_BA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                EachRunFunCalNum(Iterates)=FunCount;
                Iterates=Iterates+1;
            end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"

%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    PatternTypeTwo_BA_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    PatternTypeTwo_BA_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    PatternTypeTwo_BA_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    PatternTypeTwo_BA_result.EachRunConvergenceTime(RepeatTimes)=PatternTypeTwo_BA_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    PatternTypeTwo_BA_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"

    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [PatternTypeTwo_BA_result.Fitness.Max,PatternTypeTwo_BAFMax_Index]=max(PatternTypeTwo_BA_result.EachRunBestFitness);
    [PatternTypeTwo_BA_result.Fitness.Min,~]=min(PatternTypeTwo_BA_result.EachRunBestFitness);
    PatternTypeTwo_BA_result.Fitness.Mean=mean(PatternTypeTwo_BA_result.EachRunBestFitness);
    PatternTypeTwo_BA_result.Fitness.Variance=var(PatternTypeTwo_BA_result.EachRunBestFitness);
    PatternTypeTwo_BA_result.BestThresholds=sort(PatternTypeTwo_BA_result.EachRunBestThresholds(PatternTypeTwo_BAFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    PatternTypeTwo_BA_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    PatternTypeTwo_BA_result.MeanConvergenceTime=mean(PatternTypeTwo_BA_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存PatternTypeTwo_BA结果：ImageNmae_THChar_PatternTypeTwo_BA_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_PatternTypeTwo_BA_result.mat');
    save(FILENAME,'PatternTypeTwo_BA_result');       

disp('the Pattern Type Two Bat Algorithm is accomplised !!!')

end

function  BA_result=BA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function BA_result=BA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The BA,proposed by Xin-She Yang in 2010,is try to minimize some optimization problems
%    Calculating the maximum Kapur-Entropy/Otsu value use the Firefly Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    BA_result: is a 'Struct' containing the following result information
%        BA_result.Fitness:
%            BA_result.Fitness.Mean
%            BA_result.Fitness.Variance
%            BA_result.Fitness.Max
%            BA_result.Fitness.Min
%        BA_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        BA_result.BestThresholds: a vector containing 'TH_num' values
%        BA_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        BA_result.MeanConvergenceTime: the mean convergence time

disp('the Bat Algorithm is running...')

%% Bat Algorithm STEPS
    %% STEP ONE
        % Initialize the BA parameters
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

%% 预分配内存
BA_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
BA_result.EachRunBestFitness=zeros(1,RunTimes);
BA_result.BestThresholds=zeros(1,TH_num);
BA_result.EachRunConvergenceTime=zeros(1,RunTimes);
BA_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
BA_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
BA_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让BA跑RunTimes次
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
        %% the BA main iterations
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
                        BA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                        EachRunFunCalNum(Iterates)=FunCount;
                        Success_FindBest_Index(RepeatTimes)=1;
                        break;
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
                BA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                EachRunFunCalNum(Iterates)=FunCount;
                Iterates=Iterates+1;
            end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"

%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    BA_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    BA_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    BA_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    BA_result.EachRunConvergenceTime(RepeatTimes)=BA_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    BA_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);

end % End "for RepeatTimes=1:RunTimes"

    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [BA_result.Fitness.Max,BAFMax_Index]=max(BA_result.EachRunBestFitness);
    [BA_result.Fitness.Min,~]=min(BA_result.EachRunBestFitness);
    BA_result.Fitness.Mean=mean(BA_result.EachRunBestFitness);
    BA_result.Fitness.Variance=var(BA_result.EachRunBestFitness);
    BA_result.BestThresholds=sort(BA_result.EachRunBestThresholds(BAFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    BA_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    BA_result.MeanConvergenceTime=mean(BA_result.EachRunConvergenceTime);

disp('Statistic Metrics is Calculated !')

% 保存BA结果：ImageNmae_THChar_BA_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_BA_result.mat');
    save(FILENAME,'BA_result');       

disp('the Bat Algorithm is accomplised !!!')

end

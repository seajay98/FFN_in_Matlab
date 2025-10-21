%Make the Black Scholes Merton equation into its own function for use in
%project 2 for MTH 560.
%Destefani, 10/25/22, UD financial mathematics MS
%Prof: Dr. R. Liu.
%Due: 10/31/22


function v=BSM(x,tau,K,r,sigma,Otype)
d1=(log(x/K)+(r+sigma^2/2)*tau)/(sigma*sqrt(tau));
d2=d1-sigma*sqrt(tau);
if Otype==1 %a call
    v=x*normcdf(d1) - K*exp(-r*tau)*normcdf(d2); %Black Scholes Merton EQ for call
elseif Otype==2 %a put
    v=K*exp(-r*tau)*normcdf(-d2) - x*normcdf(-d1); %Black Scholes Merton EQ for put
end
end
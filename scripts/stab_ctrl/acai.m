function [ degree ] = acai( Bf,fcmin,fcmax,Tg )
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compute the ACAI based on K, fcmin, fcmax, and Tg
%
% By Guang-Xun Du, Quan Quan, Binxian Yang and Kai-Yuan Cai
%
% 05/05/2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%
sz=size(Bf);
n=sz(1);
m=sz(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M
M=1:m;
% index matrix S1
S1=nchoosek(M,(n-1));
% sm
sm=size(S1,1);
% fc
fc=(fcmin+fcmax)/2;
% Fc
Fc=Bf*fc;%central point
% j=1
choose=S1(1,:);
% the first row of B_(1,j)
B_1j=Bf(:,choose);%从K中选出n-1列
% set z_jk
z_jk=(fcmax-fcmin)/2;
z_jk(choose)=[];
% compute the 4 dimensona vector
kesai=null(B_1j','r');
kesai=kesai(:,1);
% kesai=sum(kesai,2);
kesai=kesai/norm(kesai);
% B_(2,j)
B_2j=Bf;
B_2j(:,choose)=[];
% compute the distance dmax from the center Fc to al the elements of the
% boundary of Omega
E=kesai'*B_2j;
dmin=zeros(sm,1);
dmax=abs(E)*z_jk;
temp=dmax-abs(kesai'*(Fc-Tg));
% Compute the distances from Tg to the boundary of Omega
dmin(1)=temp;
for j=2:sm
    choose=S1(j,:);
    B_1j=Bf(:,choose);
    z_jk=(fcmax-fcmin)/2;
    z_jk(choose)=[];
    kesai=null(B_1j','r');
    kesai=kesai(:,1);
%     kesai=sum(kesai,2);
    kesai=kesai/norm(kesai);
    B_2j=Bf;
    B_2j(:,choose)=[];
    E=kesai'*B_2j;
    dmax=abs(E)*z_jk;
    temp=dmax-abs(kesai'*(Fc-Tg));
    dmin(j)=temp;
end
if min(dmin)>=0
    degree=min(dmin);
else
    degree=-min(abs(dmin));
end
end
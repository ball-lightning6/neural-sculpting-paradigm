"""原模97任务的冻结函数对决；独立py，无需下载旧结果；支持Jupyter和续跑。"""
from __future__ import annotations
import base64, csv, hashlib, io, json, math, os, platform, signal, sys, time, types, zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Config:
    PRIMARY_CASE_IDS = (0, 1, 2)
    AUTO_FAR_IF_INCONCLUSIVE = True
    STAGE = "all"  # all / capacity / duels
    WIDTH = 128
    DEPTH = 2
    ACTIVATION = "gelu"
    TRAIN_COUNT = 6518
    DATASET_REPEATS = 2
    SEEDS_PER_DATASET = 16
    CAPACITY_SEEDS = 2
    CAPACITY_STEPS = 20000
    CAPACITY_CE = .01
    MAX_STEPS = 50000
    BATCH_SIZE = 512
    EVAL_BATCH_SIZE = 512
    EVAL_EVERY = 1000
    CHECKPOINT_EVERY = 1000
    LEARNING_RATE = .001
    ADAM_BETAS = (.9, .999)
    DATA_SEED = 202609122071
    MODEL_SEED = 202609122081
    BATCH_SEED = 202609122091
    SECONDS_PER_RUN = 1800.
    MAX_TOTAL_SECONDS = 7200.
    SNAPSHOT_STEPS = (0, 1000, 10000, 50000)
    CUDA_GRAPH = True
    DEVICE = "cuda:0"
    OUTPUT_ROOT = Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd()/"tmp"
    RUN_TAG = "v1"

FUNCTION_B85 = (
    'P)h>@EdT%j2mk;8Apm-'
    '+B%}ZT|NsC0|NjgC6aa5<WpZI|WOFWVaCrd$5CA|)00000003h>00000005k~+0wc|)UWqCi{gL^iYVX&0*WXK2;zV<PN*m<D(=euQaMOvr@o!!B*|6ehV#!'
    'Yq|bVtbgy33PyL?%=Xg6<Pv77F@%Mky|Mfrq`~UEtv`W(d*Z<z7TO|EI{<r`9-~RKz{4d^r`_KRSAOBx{^<Vzy|NbBU>i_+J`Y-'
    '?MfBf4whhn$aB701W{J+Wne=BzA#PgZJ`0_@r;a)C%W~;yIANibrV%^ugdVE&CyBmEXnzz(u8SMC0{7bPYYRB7Cy{83?gNMu078^qArV^s1c?qJRIUWCNuGs'
    'n7-'
    'K^GVBv8nQ(iE7N#cOF6_!#rc(3%~H>Z8dckiC%4_gccFZ?m7so~Oo+oSg2zXOx~{OLBuGNX0)$IZhJ@mji_Zn!VE59Dg{>h)W>cDp2?HW|M!dnP;+9c>YS4!'
    '7mmGCgbS0n0;&_#Md7=w9|vmiOg3TH7_1Bk7lA`e#ncNkP6Soa$COB{_>iE(8S1#dh3CvDTzikzL^!(#@F4D(+iumn^z}(eZr$MwP-'
    '|7ljWhAj@nMA{m34Lww3wJ;#J5U#rayArIfFcNtWB`@GwMZ^((2nmoe6P|G+vk?E}es$MIfzF1Z61;ahSe2_@=5;!kJd)qiq1|9m!k@@5-'
    'uddj~mHjN~fu&V<;@S$OFSIupJeey5;=!NoLY^0SU`B(O{x>-96NGVf-n;k770@|!Cj4SWu@vRo8<02H=UuuQ-WmziqJReP;JCk<NIq!8v?6&o30}h>~M3jw'
    '=$%w+{ipLJN#a_DvS^{Gs(@E5t(fsBwTlX<RGg+jx+Nj=-{&pSTcclF8i9*3=^K~TuS2!c<!}c-UjEGkD5f}%NxyV9#K<IW;m${&+8&E;U$7D2e(u+3UI~-'
    'HPz2_QDpXpeW7T+&Ery;&^hThXrqSN-hipBMLipzB5Jv$?!4Nay5+PPrka!BgAli27x?S`*<OC1{cvQ>@?^ZV+4l6q3?mBV+nPeBh_>E-'
    'H6%tnzHC>5O7mGDN07VU<Tb*;ISe~Z7y#TC>Z91fi}{aG#!bXJ5qzHx3mBq|H!2?a94bdu1of>8)`+UGC+O*R6j+o=?4JBLA}YQPAWu_PNj`S9viO+bT3KON'
    'S9MGfpk_JQfUn}a`x=%NlZV^re)+1dPh$$oA$!1V@r=5*n=y-'
    '|0lwRVx;%CPm$6nY%oRz2?zH2Wib^cb1Zqaxq&Kn0{@V3WnBLUZ@2?c2G@I=|~JIxy0wkz=G#$=>}3>NfrMreyHb93sbM{g4Tb@1yi1)|hl!*r1<KxB`3KSI'
    'hIzUnNybYV0zaXVbaO+@!nJKoVW4<@`POPz<<yAmJLqGI-'
    '?Y{n5gX*7QXoG~XT|5_@3OIgo$6WHLVgsJ<s2qI)*R^)gbu+z6d3S?=P=1&w|rGr?RQLyMP9)52to*V*T&ltDkAUNoS!I`3+`|0g=>cRmSI5bIA%ksIyf^ky'
    'nRR*`Il%H4wdbQ%~K3x4Y`v*qaJ-VbD|(l=ju+8uPz#JD<XzKZ*ekKUZC&q(rPh5K}`v`xO5pl{_0e(c;b4(Yiqhj{FA+C(-'
    '<w@dX~m11(OphDf9_6n4e48X=GonVLEj4@36cK62CU)2Xn)hIu%_qv--!Pjri_>0X_cmh2w7qQFkSZdiU9k!nxV^$bN8joD3-'
    '46S6T_QD|#A`vyfmO_!`jm^G#+y{kYLylx4jAOc`KL0Mx7|A}e+#L>Q5>wpi+zNz-'
    '$bZOVa?#_Tw_9JAo>XJtdKuDyHNYvrt~<U7e{+_zeMW?KRyK1>dN^F{1{pEMx7Vl<GPKVa3Yp1qcSPu$Zrd8ZrT3nKt_^~Re^H#9I=_SuYPb|%Z}%%TD}*1$'
    'WJ4N5v7%K3L8BW-%8PQG@1zDFewqoV67G<hGEaG7l4wyA5A7$#W-$a&2M^{-ml0TZ_heq(q-(yD=}Y{UwyQI=6g%!aecN4-'
    '?GNtxzDrS+D<zAe}MmSb03pCiQ@Sypg{e4XCiyZfRIZut=K2Om~9R+>BfFKLl(DMh`+p5&+`NV!{+s|lFVNL|1OriH<-'
    '==hF|^I>l4BE4}+?7=gqNoWWDq%8!z|gbB4Wt$Io6c{>KH+ws6$mZz=nFrI}L&&fmM^b0U`3S~;B@QC(;H!#WR8hj<g#&HK2))Ou_q7%oL(0UA+U>n?^8TmQ'
    '+r;Hy9&P<VBY7cFK~QV<p~vghryx(Uah@mJ4x1B>;3-;CXV=wF*_OcU<``hE_XLoweUzQfn8xgY@*P7rhUdlN1xqR@r2d-'
    'b_3^Yv~dx?Lb=6YA=L$Zp@9bFtm`y{H%8c5<{0kY(X`lBd>W8_Aa34cv<71L*_aT5Y2tB~SJG{bMq2Y*JgkvmJ%5T262J(}8R$SvDWyFR}h4kdvSgvr<5n3S'
    '6~TRGGZHg~YNvPF9mCDO=^G^A=A7z8l(%eUItX5!mI#n;~Z!{U!f1h@k2iQ9gg4XxaK|a$F=Q@AJ7Gw3mhu>prjm@|omT^FT2=nFR|~`LU3+aXA)x?5?@lH;'
    '3T9xDG7DmSk#%GymN~(!BB73ad--nLi6(C0#h5hf`nr9Zr%@>2*zgFV)oH*4_=++V!uO3cowY_?~L~@(?Y$qkikK6Ccd}d<k=f>r|*m(L0b|d^?}&@Co(K@}'
    '`X3k<m1gp;F`mHGtz#O;2{aOd;UWm+S^&TJ2}*y)VtpTP6ap?3;@`qIKj#thF-'
    '3mNy5p={(vtE?cj1H2Un4=y|`82Q~a^R<>Mp8h_M<NmN0!?{`I~Y>58Yp=x1$F9nXsPkk@b7<l*pfUkFd9qDvcE&NzRT#nDHTdeS&f$d?^oS<jlvXLDTx3Rd'
    '-X{_^BD!Q}0c1EjaPyjnt{T@AXw-4qo^d7tQ?jf~F2qtHuxtd;GN?Qd1bP$EAWC#2<&_K?sMvz7>gq^7fyPgBYbt!1e`S#M9{lVtQj-'
    'B$r;AeZr<Z7KSnF`Zb$2ABnj@-'
    'p2S7Xwb4>zCxbvGhrT`C=LfmQg*F?g&{nGTf7Tq9dDa88|~y>fbXIZV;gnE(r9qjD@R@4T1(HkdeDJ)k~*+?~9oskHM*v_E~%8hAfdp3axf@rXk+jj-lrV*p'
    'nZ>`P;U-||_$!M>z`-L`2StlPC|7VQB|1-6&x0zywXe3=-'
    '8`$+4Qod<>|bd|{La`phfU(sXPmuvONh^VV&Yoh9h^rqw$C%Rq@hQGN`cZ^coJ9|uLC9_dqi|>y`)=$eYJDN@xFYnC4P^3N8=_f|M7PlwoCD}^b>1Q1&Gvb5'
    '+G`9|ivgbn4-FG%W9zSmGFOOv_cDOP=7{5yECM-{#C*_Ml`Hb9I2b<h}v)N=0-$w}E--6LR0^pgYpL=|ANSXQ;&4F2I>fH8b?vcPfOGkR}aSSYotY>t0tB-d'
    'GB<HKza7dcdAs*L^(*SD2w-gsOj_*mODwt}0pU?{m5}8E_No+(`$o>NcL2)!2|3*3Qc&L7~Yam)Y)ngyx3R|nzGfiyTBd|2{5^MaF;~0U4`-'
    '>3Ia6UM~_VZ&9URdm{uMNT4gweDxyzYPXt_DuRhlLz%cjRDc2fVY$FOCQH_(IJz&8&&-0{48LuT{xXxI=6&(DvqYE;PsIrmfv>@jHeNOn;)Y-'
    '2KkUu@;O%%>Wd%8V2KG3Oyb6{g+|{kxRh^ib@J>=Hzq`X(hF)6um_|NY8Fh&RzWSw`DsOgjU<XkPGFcrLtkCzmCR>iBaFjCm57#<V7aasxFy!vX?sgb!>gyr'
    'uL`p<g5?DN!7hAWNut(XFk#L@ckGDeB){u$71Vqw&#1Us$i|OXw-?j4L3UKOb=c-'
    'Grm~dMSH0Z<xsA4D8Jz0deO}Gk~c`6hq8xH)(75Fo!UBC!=wllcwaBWvMMIF&r+o941w|o!4J@3RWTa-7x}2n&v=eK*7-'
    'EDL&jcvVM05xMOlqw#vf8zv1;ZyP_ALj7ws3**Wb7(sL4PUj3uUfZX;K;lChoc@2&J@zu_7Mb$4jN{}6Djg-'
    '1;4OZst{(0z6<+X`Pl=Xa{&U2dg!Ph}c{a1HsAsk86by8xT3a(d17-bF8eBhyH8_To34=pj5=D51-'
    'h8v#mr{WX_UH?zoICyJZ+g8M`Y=@s$<y~}o?aE|*9WHbYCpA5XZdSak!$tkm6DX!}217M%ydUAU4wWjefEB~b~i?VAuK{*B&^v^v${-'
    '+#sE<jhfl@r*qAYrJ|@KyhF+jX&O_I=mwE}$e7lc)gU*2X+x;_H_ARw(!66|XApI!U{)1NQWmqur-J*%v>J3BTQ=$wdmma-'
    '@4iazre1cj@QLSC6Lo0)NX@ec5<QYy%X!Qk<YSP6d)ph738C5tLY#JK@~sC4pe6S)E|{h4b98hf^8vSrXmSanu>1`#|06=}azMh|j+9CnJW+;osDWEsduWnr'
    '>$;Ip4p$v~frCC#lc-yE?V6t@5DA?I2{7wSo!gCc(qgcUvuV=t5K~wwcbiBH{SE))v(C>>v8+5H_NkSmEdkOkTv6L%XBj#3>Bp6=ToLg>s1Uy_3gIQ$89>$1'
    'Zl$933c}POg|UjVE3EEzThzBKz(We`gh3;J2Ed2a7}Hy#HnUXX_WW*F7d)D8hJS@f21BSFUoDlxs!GkGCZ_K}Q;kpVJ(@#d`skvY}73p#OThdO7Tmh2!47zZ'
    '1WcXKfbNUWfns>pnu>)ObG)SDB$cUDSdbeC3_S`xMtd6K7Mr2RB{`6vyoH!3SES4Xw_FT<$RP#WQxIL7qOvo;HiWKfk?X0FP5^DTT5fe@!XiFZtQ3S)ModL7'
    '`3X@ESB_xQrfB^IDpV5H;p~7sk=KUk&l+8xTs4Z_QwRbBBwienwl8axj_GYB{Ow)BVF}_MJ?n+vs`*lHP1bU&I&IAK6ZH3_e0C5mL{&jGMg1B1L#!34t$vd~'
    'JC5`ZpA_dT6wK<k)jB6DK<D0G9i+Hw7eQ2LBAexr)o0<=&Sq2e9s&Y6#T7{gY-'
    '7r+_GV84&4tgUn<seSbNjlIyLCtv<!N!|h<MA;FAy9r0~^7vI0ze|TyAPtpr^ob!jPJ!qWiY^o-'
    'M^>bv`j!L8MQQFarU>jeFx3ZIuhp;A=sV<N>IH<&zzWkC&?A{c&&d%<e)5xW=<yPXo8$R5EpJ7Qyz7MvOqW6gH5_NOZi6^T07|Y&4Bli3q+x|po9PiAyQ*wT'
    'OZ+Whuo)$}QF^R4A^!K&%{{|j7t`FvZV*)2W9xR*wn8^)RV5(ovIB5M{9C!JAw9mtT>jJKpT-'
    'h7W`RYpxR`PH6()i{lE3Y5Uw|@QxRjX2|ZPTINx8G#Gh{*_@;=+TqZyRcX%I5Nvcp=beh4?*F$NAa|&Ykt>Z2C&j-'
    '*xo5ZmlE41}#C$K`WzwYxz=qVwTYG@jQ<70@uT}D#qH|7lOFl51l51Jxci8<*w4JmYuh;Xq}z5nvy@$!JtJB?Qj4$BYSf{{@B<2?31+)*7G<m9!B@WTwHC7Q'
    '&tFLvEd?B`CSsbty>x^@deqgEtq;K{k-mehPhRsT{(-'
    'dA?}b=km4APeeNS?tXF>0Ox9!jkgPn+%md7}7tLv@`kd|Ky?2Q&=g@xCw`~X7#meBvp5khK3gOT1L$rTBiSR#Tw|I-'
    '{!<X3K9hoGnUOsmvtW5s6;ayJ~IhkCCj9OLy7}w&F-47!MFQanK3BcU8{Kuh_?7!l2OiXqLBWcL$(eZLl!fw1J#^V-OQ$8|0E-'
    'PqDnSag`Uu|>cv$6UEt_f(40&4%B|2oTH#Aqi{1?*?mGu)_g%rjEQ-'
    '{rjD?$B$rZ=PO{47$_+PF>bpstX=Q!D;^mS{yg4os{t;EC(D9#cT**0`SgjqUG?hdeOeoZ%tZD$&hy-'
    'u8}3&LD*$5h|QFSI~lT#5C?`TH7OGR(v91>9b1XUnroH|astpAX8TyK0fDkVx?-v^L|p=X`WNQ{YkGZ(Y(<-#znTycZ-sFPwJ`%+*1h`V#RU0nBVXUt!~L)M'
    '+C0X_Rj=<}zn>WTHFVJ)bkWzP0M`bg6d0_nuEA%uxhVq`rLb%FCih9bF`kEA!k@#+hhs7IO=0j1K25>#HZ}T+u0A1T3e`H6nJhWlXuD_q+o0Kq7xMFWr7>w>'
    'k=1dqQD(n<+|i8^Yj1<Id)f_ey)bc5jqT<uA!POViU~c_PXF{hu-'
    'z2N$^IkyiH*==%H{Ckn&lGsdkS_t+uvgFp|=oHtt=lik4!CYOzq8)^dSVsFIm2JUp#K}bBmzqQO_?TXQ59HMzQeWeOEKZ<8=|JPWgxYyM?N)PUDiKl9dy&fH'
    'v*Ds>V9W_Sy!$q}BATQa?&~*T}#M&|!c4l=rqu9DJ2uDABFl>RFd3jQ0{9+jIIAg*WVaRZCqrgYoahOn25fmx~-Lq3+}{d-'
    '^d3EY*0({H^qE@I<Hlf4k`wSG*`#A;Oe-EB1=>n5t;6oR}zBHuOWUg?;zE&q$R>Af@B~bhRubjPuD6=fh`B6T?wwvus<ZhhSatl-'
    '!InmllWdA*c>UU){;i*DVB<`x~>xP6s)r?RF(1>HH%oZ1Y<g0LbI2Oitncy;6rNsGdku9e;EdX=!$U>YpP*d>Sh$!RmRwJzng8YJ3whAGhrXujKEUTLS;ueq'
    'X6fjU4%mr!Oh53l3fhpXo`m9ll$>bba<jR`EM}zHQ-NEz|dyLzpdum<F=h<KpAiTg6<13(bnJLCAV!G8jMfjlnAspK(iv85L@<2wPAyyeaqV!5t<K7SxOq2n'
    'O3|J0^Y~hs$Hf@UzO)M&6CEDFDI0#J+qSaFx^UH#rr3@U%H<jUoGjku^&R;{{X+2HbpwZE>g6bJ>LR9q(59RxUaIRH!_(Uyo1|*(qIz>0mjS+620h-'
    'g?S>`DE7a@};ldk4Jv{6(F}y!-'
    'BKj4?k_zYR|><sW$0ES>zwq|021;03>5XT8o8PYL$Qg0SPUPa3gcYt=sol;X`rnRjG$`tD|S;ocT!}|2C`qTJ5|WGgCYPzdX6`JWswod`h?8ywg73CpBQH??'
    '|?(rOj_xwKqR_X8Qw@t?f&<);{~F!Q0xC&c07Qjf|)PWf%uaL!jnXDkHg_tqrDXSv4BXzy(>gJ?s7OFCvWtwd*2dD%1X_N!4BF{HfL2sngX_$E*3|Y<qwad9'
    '(l?!&8mJ+Bfi@mkNWs?%7i_{;BYD=du<2IGDgr_sxPQq1sY-?$3(8U3HpBrU&)5MsDG&<R|4_-B_B)t=7>L-'
    'v^LzLYUQ52F_O()lRk*jiu1_3Mt68;#>Gw=&X&V%lJUyqw3<)I9Ex2B;5aQuyf9IW*POQ9(RSztb7=}LyZ%9smSDhH;2?mk&n8c>#LdubM6n*@YO(p)0M%ZX'
    '=eV(b2V9UYJoT~ilXuRu9wJG)RlcX_O9c!PmN=4Xp`O2Ls~l$_wvl;QuXFdiD1lbgiMzg!1Fd<10Q*TNg{QU!(ZA2V{iCgl4#D>N2`DuosLT_VPo$aY%p3gG'
    'GBP+25gE+J|qe`Y!9DMp6R;r(qFC8V-|h4&=QNAQm=l>SVrkWL~wiW4O^DJOi>5l(B~_Ei=ppMYKB((-'
    'u4?=XP5Wtf*Pd?F*vEWDzjT)SN~gKhX&OFq}6a4`u?HiOtT&e1Ty=UBre9`JF^6nD<g1Yi*a-R`9w;U-V6nt%y)9oBPIGq5|vzh#R&LRPrg|qSKVRG`-'
    '^u0B~-T}SjpaXmb<=Iyc;X@Tg?wdoRN(Y5O=5V!|U<d&-H#|3|Gi%dj&f?+o(BFH}iT`Ce=k}J#G8$igRryH3-'
    'sB%t4#>PI7oO5He~XgG_(4!Sy`ms(8{V_tLU>RB!$YWMm9~I}@F%Vtqyw+{H99&b>x3cUBCC{M{w2T@~L*?^ByM$ij!^<$C9&M>e&_<zc3ifqQTi@FJ7vXy|'
    '%mW2&qfBL|h`oinSL3IA;447L?tHJQAU7a8Ol&(C-m5oPIkND0Fwbbm&3me!3qR2c5U{U)*uEW_sXAhhmDk$!br-'
    'c4KQZR*dTjLA|nzHdWcq*O{rNwfU7NO>jPjOK&W({RuQAYX9Vtk{U}a31|MFuslx;wPCeqDX2y^oR;yPUq)vq4r|&4pOOM6Xwk1saXfDrmRgS_U-'
    '?KjW5Nu<3rE?s3tBG3CXIlNemqQhV9LOLfLAz-'
    ';=k(>GgC{gJJOT_cZFO^MjvF&GFPM+3X$wNm`IlX2ZyuQUWT`V4<x5ar`*1E%F9w(2>}W311_n=ssdt^td6t`z2@moK}tmZCrElXe5<&nA_hFjc?8ge59t@m'
    '>R5{PMPC_%zSFI4BzUNVq9<u!qNl52s+1>*jgxZ3TPi-'
    '4NoQ8eHUkknA(j!?|{A6FMsn0eLjo>mky2P@lbSjO}G0s@T%<lcJk|oioXL$7gEtcFDlgl4(tPD;<juLx_+W~n~u2e+bO2XyXf5OFy<+=FKESv3jS)qP;QO#'
    '$bGyKdUkDpDkN=!s$Z?g_f|TA@E0!Hr>FlVw1{AAO3JyAz!EL|aMo^R)$%Xq(S1kx212JLsjcC7UJO-'
    'A<~@9vw#Dw%^ta|Yrhni{0AfrU8=4^d*9_Y|b(Q*g?Qfl!=XhHppz===#`6omnbGqex)+nnYk6fsTziBdAEh2_ic@SJoZSrHgRYPCZ!z5bJuZz4M~HxS3f^w'
    '{O5uGwCqjO4O=s4}%#6>qi@Ef++3sG)Z)NQxq4dCACp@10*8M)TH>HB4H9F4l_B2e8tv6AZRVRv<L%qBHFF5W4996+jLj4s(l56tXCo=wKYSz&pXN4=ftQ%e'
    'pgWGYa)2BdHJ;_ewcn(R-`mvmJBsys+{-'
    'O*8OHMBhDH%{W#25Cv9&&)pFG&R*vkY@c=vdrrYo%^2s|{DfjZ)fcPvl!ILH$3%hc<lPceNDeKH@i9wX%gezllb=0j3=f3^;YYk*-'
    'hu9sJ!&p=Atl2L66~*b4vwec8EZy}Cab(}U7*SJ-xVqwm{kT(YQTILQv&8u8>ZkbZ#ex!a^?0Nrd|)}h3vZiOXWyET_qHjvF;US|L|bT^ZYoj1q6uTHlBH0G'
    '9*gDUs!M4QQ5q-_ZWY10x5163~Fa^<HN;yeIqa5XdTVf`I5^IogJOcgGQRaz^^>oy&b_vk3hUtd{>FWz#&I-'
    'I{m)Z07r#)d*rs5af|+ParfsMNvzF8X5Kx=aKMSnGgu52uD6?q&qMr~{@)HQEhhx;wu@kFMF`>D$^4Q`O2dh<7)V+d88(H;l=b<sjFcn)4Hq6=3s+tH2t)xn'
    ';@Kn3c$9WpyL<o7)V|Mq2yh#GGvr?3>8En{dqz8pdQ3GK+(I;ec)vZ&|G1n_apHmPHUR0)^~$o|>>B>=R_M@a_k+Uq|(r)yHAF8rU{K#CmJ1DzH7u_pLlC2d'
    '`1{vTNy$eK7!Tc9E}E3gHO=KI9L~CaXT|b!KdI&)q#x?BYxm|B}eTTwlpk`vB~pdDK54>O$su2-'
    '2k^Hr?e<)D{z3pQ<APC*U&ouN(kry~cl|yG8?T>UA>DGup{T60c<~lN9|g_1m7+sT8HZeRKEF{U_(W{`>bA@<-<5w#;5e{6Y&Bj@8a%5ii>1iuv@-'
    '1dnb(?4@0a&Z+Zzos2N)=7n1hqR2WBnI?RFuo{msug1n5bz|B{02l=vhg8&;$li|BsWDG~a*a=BiY^ATwfY7>nY})mYZKLC`1ERnsL%`atwYvw9rERzR9>A>'
    'NPFBVzTngqc#zD-(N@CqG81<L-ET%O=O<6fwP`(AJk{;CV^gtx%*!^(Qhxr-th2z(-'
    '<=_A+3x1&<=AXM1~c`~b!wFZm>y#B<rL{eK6QymKxD$6<IiJ@{vq{5{}Cds$x#|A7qyo=U~3Hb%lx7A`-b%oT9s}-'
    '>DC#)pORAy%7@eU9b4}!zYjLPN(bV1UzzZtNWWnbh#8ks=2~G|{b#%2c5<cWV$;Cx{Yc&nr~KXmfqSj)EB7kG!NCX0Yg8LPgk)y;D=>evjMD8_N}s;V#F0RC'
    'wXdQ!%~c!otNlz1XM0rGKxSv%D&Alr)x*?7K-~DP%91+w6P-)+Mv8r$DL`69Y9v-nb6BbV`s=$RkkexG5?zFub~!g;SAwtYmiNa};-'
    'OjSGUjdTm|IG5dumecTlo0h$zQ^L4F$8mfN>@}zxb_djCB9++V0(gN^6_0hqOf^x?z0y8Ay=%YQF7XW=`2_fe^YD`@Nv!HQK30v8$;3{lY+g_ytzc;j6^Y_%'
    'r6eJr~7(obJ>%{bHP(3UW@mT^<>wKUXRd@b15^i;>QZxgEst_t4v-!S!#gzsZ)x2G+3S+ASoIn(F_)O99-'
    '}w)kc!tYhdUx8VW6GZ|FTvIN*mTzC3@`Od|qp?`20*_yd!YL6$rOr}2Ro2w7bqh03ql{1Otj1KO)L2=vRiqQQRsHFb^Kmgzpro4xx<_aoOWB&)L33)Oz$!0V'
    '4f`Eb0rJj&qko&Ej9s_b5^DnrSBn{d?ZEK^WqulR3JB1Gq7LH}btiOkRkmh!?^z27}N&pK6wYuKsE#&*dX7}Qy9~`i{TuDW^>&AM|1AS#>r_y}j$zE4KpmE#'
    'Z6i#%yv~nN#Wpy>N4t)RC+tr=GXfSb>RDRg~Q#0EyI;y-'
    '&4U4~YAt0qwy<zi_+})6NOVGE&IC>0(L4o=6fYNx}Z7Qi^4qUJKi{62}*YUWErE)$@+V&TT1xuZ$W_NIE=7MVtzMWET&z*2r7H%o8A6F_@R`B!Q-'
    '{JoIJ4zM}cib!P0>ru&Q?~0cid&!UcpF{*@%B=m|2<NW{jTg3ay!#fntU_XuOobfa>h0|ocR$pQ#z$awr}<ro{b_D&CnT2zP!2m{cKo_pCNqBz-'
    'I0jzKhdYdT2niNi3~d<^&Y5Y9bN#ZBJ)0vdF_o8}C641f4C`>qj3-'
    '7Dh&x_5G^D^yiX8`CTtht;_wu>uuK9!ELe|SDQQ=KdKjJ`*|vuuQP`TgOTeesu~kU!m@H5vB55M3Rt4bjV~~sU$}}NWUowY9_O=9hY;{el2{!_*b9g-'
    '70`vbjBpyJ-Og*>Y=39@ZS>hXAX<YJocOJs?{7lf;6U%^oj|R>(7#p*L7m9#)ZEGOShNC-RzfL9%h%H-_+`(kn|&7`bo6Pce-'
    'u6ut?|&Q%MB@^u%}~)n4`mR?a=S`;uTjZUP?DS@CZF;8u3YiFkw~Q9@UCG_){;~#X`5+ZFYR1`TRTmFM=G5&gj<c=#S|<-'
    '9H8!soDHqE}t&OuZNGj_B0wV)*t5d8UPnR8u+XRZLQ$ZeDFPf0il_rM?8W%@+-y#CY%%OvrhG1y_yRB4NX$76g{lzCX#-'
    '=XM>m%`na((R}Z(`(xwWqvxFJye;wk~y}1umoq)R#;(H{T$NDoooK=Uh&~k}{iuzNbE>}vWp5i1tFOfsC`sC7zr`taNtl4LT?n7^UFS~@0VHQG+&woZ(q@j{'
    'M-YWK`-@8JblZF->-~Pd3)mXJtnSnT6H#3jIUsURXvu=^Lv*th`vL_({j(KPsoC}S}@IX9<$Mw4YsS^FLaj%rDqYFvJ#Hh_~by-9E?{!a&Q^h5-'
    'aF%uMR76ptGr2t91X2*A0n|wC2eEzWH|hH4kpDsCMn*`bnBBxK6^S!&7@JDYo1{ae=tO?dr4TgcnOHwYLftl$p2i*vTt8d<tfA)!7<znF`2AD96kUFgiFv7T'
    '4w}cJQ63?WLY-MV;d-VLkYYE>`VB?289pV#DY9~B^X<?v6APAab#%0xAmHlk-XFoCP^>{e#-*CIZD_VWG|IMt$LN>;-'
    'f|`b30He!D9&q>=mdCPl|dnKH`RSBkB4*F6;*#x(m;-#j?4`4r>Gyy|4D=Fpxh9|6lflQzet}Yk0-NnT?3K%efCl>V0i!Yr&hDx6`Ai|)-'
    'Jn%9v#}+Auh?3er108?4M)P<ml=Guo4yI;mK%o`Z4!JpHE(_2{!?`Rhmyon5orz4u~Cxx#L5yc40R4k+#b%xGNT$naf4#meN-'
    '8a<li*c=HGvRV7B`vqht1fyHyA;7X+@G`xYs>UqF)Kg@dU?(3{qX|oJ6W~>1&B@pNC%Z(VTz$e1asOHG$bU14=z0OJ-'
    '&VL{3v*Vg&i~GiN0awVjQ`NjL6^B9ks7(5U(IM4Kj6O?uNq(qhun}gfJ?<tzEgxK<a8p)^OS6m}YWr?_y(}ksnM$ki(SxDr=lYAojw85vosA_*jr$-'
    'lu>52!-'
    '8Aj|`Ncx7bMPnQ?PZnCuDfO^T^ZJgpn(Os$0fi_Qh{O>eDmF}(sy9luAFT9U=fpp3rcA*2s1KOU)JiO^XvD#ylVG=v$D`_zOtviqmDwukwh@Bd~$zIA6MgSG'
    'cxIg8O50%$Ejbxv=|)<$ICfPs>AJa=jV6JhWByK*QQ0sYBw+p@6~?9mpX7(=(juNZoqhJ)A=ZP9gMXH&^8+V;<)$OJSaKh77XCN*r~niBfnRCq3x6-'
    '4(9;hT=sOX6n9va@1zHny#dy{rx~QOSvVxG%*=RLueG@6_MDA#`Rz4h9V!Zz`@B+0SCY_4O6c|;=Vu|PWBP7Z;^540`R~9+a?@Ag_GETwk6VXjX0o`xe8u_m'
    '?A~_j$XJaSA_**C@!WXyH(rK-6<ChftCg1hGKcG0cM<~Ao!EYAUUNe`nHmD8-'
    'u`_Bo`Yl*pv0PsKf!6H^o?CMD=1%JsP3<7Y5R^7kAx$CU^W63eoowuXFsK9c|9sGvDzxkKi~eJsW7#e)uN*jxF3{_eK_uG_(ffr@7+c*|C>vbMetr|+`3XSf'
    '<^T1ftp%0V=dBDKu5A<_hJ^o{j}EWA-w9;{b#KeK%nbK^dBM-Cyo8&KgW}`1Cuiz0pIQGPsav<%_v%gsK~&)dw%*+&OVIIWYA5d2I1+c4<)Yaw6d;{d_%O$P'
    'qK>b`Ohd^B)51`QKd|_*UU#>ox^4N#j<|dImHfJ<(ROl%~$v${e4zFSbfoh6618LaUvE^B3C@(j|Osn=ISU=y101n-'
    'UvRoi$QotV3&G@Z7+O<Td1IjlUJ_kyCjB%aM=`$%-}D)zQwyef#vnM5Di>k0r83lJLf8Y7}i&onyF{C^vUt#(fRc+7?>Aq>O0K-'
    '#i7m;^{&3geP^~K=B9$^R9o(q>6rRh&t{4JsK})p=sAi!DXew@Xpun8==|G_fY5W1%Qp7^qDyl<8XJLInVqlS(g$@1iC9rv49D3;L!Y-'
    'mxxt2SRr!7zyhqI@NS?!%vBI9WM=U=bM7Z!d*VyNgfg&Zy>M@_X<grt-TN$U(H-'
    'I<0ZTTGkRJ&1)`CKkcb~kBGni6?v(7i7p4AP1|O>a#ugmIpOCEfO6GG7erOO<!_L|{=Ufq{oX8OG}h2By`qgg)8$w%6@jb1`ZTO?e?#eIgkzbH~%#^$Grdq9'
    '344RMcFo*8D;&t%k2&@%#8yw||=u{Y;jjt!$dU)3xJQvf;HC+r01k-tT{r^T&t06Q{fVq0wk;xzBFt)tP?^IF*>J-'
    'cH#bHyR#y+jV8NHQ(d)I=DB#rL8^y<d5&z3L+F4e%__<qcShf3kr>s*fu-lXLoGK^bD8`dAJv|ZP{0tP&8OEVzod@PbCivk8JhB(E;60#k73H`s=8{(M4=R`'
    'y)b?2kwT7r#Yy+$!B+4Z_2sl6_ZSh`YhnzDbOoE^=gMzaT`sw7HpVFgNX{wJV33}m;J^2Oj>u`@gN==1k-'
    'vgm`m`3SPw*Ziw(GOSzuSx(xa))A{n!h&Jc$$G-*V-Y$46HQrQ)_I@szLnP>lmQ+X2nX4K&?ljpwC)h8JC7rJ$EJN6;o-'
    'Pc`EFBgwBCJ|X6x#=U(n3R^qD_>dEFUQ=xl6HKV`a+Aqz#*Cc_EOEFrt|YV-'
    '?^`ngZ;RFq4a81J$4n`FZxfT0S8}xmQ_TWcosXYDcX&kf~I@Z%|dj${Jg!@?05JJh(0;n-'
    '>5z~&|O};+0(P*<r=$B>VlkF%IJ0QmF|#v@NW*r%}aNGk8FA^52<N>X{%Fx{V_j-QfImB4nj40@e5X_R|900u>A@$i$zSDme%R+HvT$>qW&;mcv#JJ#qznQG'
    ';W;6@pYHeLhID>R}8!Q^k5~<mC0fojZ6$_mNIT)`16>fB#$!m2PYmD)eH4Qw%x5s9O%r6t~&8;Zn2oO0h{?%eR~+6dac)P!;JexYnWbKvQX=@2+cQPY;Rr4`'
    'tZY;LQU>cUeSGVHSNUyE#KV>_OuY&E!1n#GA5?F$wcz*9VhAEzWn|>^s9UrMC<KUni(95_dQh#y>yII7`5>8%GM_1i?#h_fL*;Jq5R_ylHvuCo3vlG@B^RRX'
    '1@I_`Rbh#-'
    'vxlo=tQ_ZXpU%OnL*$QyMr>Px3G{sd9w81Q*TNS|6pqzTf~DMZdEvmW+Ot)KJH8jkQJ`8)K(Y*Cx<e~KDgJN6g#Ph5?u80*i|}4Sue>xwx8>*2Ua^CP4+US+'
    'Z@Av;q+<w9qQveP)_G|`c>9n(&&9vzgm&dT>5P8Dv92tT<p|Ot5ivi;J_|<=>noIfuC;DD>PPpzl0yt1QNM?ETSzWy7y9ZEo8*RGuqDW1$fvVZN?AYS6GxL@'
    'L8Ga+7W$ilj?lr*1XYc^2eCKx2gV+S~!!BI5d(vWQQ5ArI&^Y0L!hT9Bh<E2lgr#ifVfm%O_I!HJm3XzmtUIgQVlP@Vf1#basiMwY(f_c5eI0*EjEOs$yE{@'
    'qT5S&EI3=L{wD1)2waEK3n=-({zf-hA!m2S-'
    'u+SbnoTVijCWN0ONbOxqAl+0{#)QxD$8H@V!^Q<c@202&maz$z82N$!ESdK%upqI@fnpHsK)2Q{FnkDuY(;lpA$kkNEOAMLzO`jWYPcZjiZf?RhZm@}$v2x('
    'Byy&W-'
    'Z89r@({uEzZmGQSB5o%uS8JkKXHL&EANJ3Ts6MLlhP_V3?@d96!q_7F*@RD7M;_&cCKbx)jfsV<$w5wKXBCw^yN@4~zQo@%r|#Wg79kIlP#==$@?D+$&Ii9B'
    'Go^Er4;EZ|?FTc)Ct?yPMlaoCPz_ra|l^6o2s(vN-fs<&@aBU>B(8_xSYljOB^y`|9bB^~RwtnYkvOn4va^mk;o(ay@rChx4O`5r6g5potqwgRuX2suwCjs9'
    '`m{FYZdJ6Zgwv`|RrU_si~fxnUwy`1Um%zk?P0=9ZQi#$%#teQIHx2;$rV5aZfCDI(NC(BpK-jMcUJ(k>JauJAeW#T!>EIwIj)fM(b49|h5sZ4c)+2np!m+l'
    '*Evo5SUCLf99pHy>X#XObk9yIHy`@`?_ybndZtg?gug&0!^MN1UX8uvSKF<k$^Gjihc%eegU-#U5sms<FrQ`8EJZ6&<EpM7-'
    's6YfrEg5oD55;zaFmz>=ls+V;7pU}<Kt1qg1R0##aG*Vd~#DqFlrT3nE93QIJiD>bMI4sg%afXem+&z}@2Ej=Vt0jLSJ93}&nsMlon0oVQfk;9{`EzIv2Zvb'
    'I{$``%X3*bFe{Nyy{IJk4&MSa_CM3C*X?Q>jUI^54cY*@vU23DnDY{Lx8pJa^>g9Ywy1A@{_|$M^WU)yft)Ek0snu)$v0xuhV$g+M82-'
    '?Z)5yK{Y@0q!c+H~SU4iH=4(9L?8Pwnry_{w9BhvCy2Pbl$j+ceYwxQr#E10fnZJx-li%d3_O3JLb1Pq%?t))o{2)3<Gw6~N7?Lw#bTi*81PRU-pU%-'
    'X0+#~l74HeCQfjXWp4H*6oT+3SVohwd9_xLp}me!NB+Wjn&b10qBwtB!9OX`2;KIY7QeekDb2g{|So#!C3X^#ExTBKF~OKNH9cWA;TEhK*HyKXjjuWVYVFd('
    'NAxwUfOW4VeHrvG2t%~F=iu$q@g7ppE|l|sSp+7cJ6l`Hu&vD#Qs-etY;84^lxyvhW|8L+DDwA3?M7)qrVM_)0ZnkmO{ax${YV6<T@42+Df%j6RMWn?$=rwy'
    'MB;Bc>Gz47xg<R7N5Mj63GH%}m%9CmJZtm|_LUb!8OZlxv04y?BEZj^snu74AQ!nDj{&vjKtAHP=I{~GR0bv=Ani&$glEA=ZvjVH77Q$GCN{0(qj>R8L;Ev~'
    'R6HTU6K_u(1yw1EK1^1WtGe~K5TDH|A0x+8!6EA5+kr++>OjdCuwnRdiZ6uv+nG~LBVv91<$(e1kA9}lOzJ-'
    ';y*c_~e;hB4UJ7a;o{%VTf9?lObx?vj(wyCm4@p{Zrn0F69Bv@dw+y@7&5;qx{5^Elp)*wvX`YAoT(WpZ0-'
    '<wGLiy67YQy3@l?UC(Iv`u>;{n&&doIJ3dmOMI$N(1X6`E1KL283Bx9ua;CIzNN)?S@=|n|KLu6HTRlaIUoou+LW`P?TMXncg+xxKG-|Q+V-'
    ';Hn7o$)uob)3lT8&K_jyzf3u&QgeACScbP0t`Per;CGn$2FeN?PQptm<>GNe!RhIqpk$xWh{NVJpVN@cVt1`gh~mr7M){)`Xg#dq%)js-Nk4I)D<nxmrkV^%'
    ')>DoIx#xk_G&-tOzRWVJiNA!@B}!(_VkeaBM;--sR5#r3lid_`&^v7QFJZuoJGO{>#qY1`sWNv-'
    '*`$4WX8Q9{inOdT<1;mWI{PkMg%4JU$fyip>OsiM{&$$fd2kY9R}t-rjbn{N)hRx9+@i6F;iVeOG;j2fkGckb(UxQNc}$!-'
    'v)+)#^23CkPpnomv+ALSAX2WqFiTl#A)%H!!3(oe_EQ)>+-WMo9ac0IFazWBo{He03eIEK)d@zeD$V5mQyPuWEq|Ew340ap(DTG48$-'
    'V{DUp$zhfNgc`g>Cw6koD;6N4>l0|?EN;hlE-~RyGv`Qgn2tVPA1q~rMZ{t-'
    '2iP~<Fnm{aQgebIHvUNYMq$Jp_p1tU<R@e7C*^#+S`wRS*0`SgXi8not_3yqb*TZVo@p~1VoTII-f-'
    'RsMLCiq|)dN)14a|0p6m;qC<)|*i5&*PDapq?yU8qiLs3~vXds}BiGy{Km-zlE$QSv;q^V*)0Tf!N_<@Lf>#c5^W*H}al+i(v0cH+V|**=X~~d25q9})mH(D'
    'gSq4k#9)=`}K-9gH!wQDA$#*+38>#}E46ECM`r_4GKhI{xnvJo`{PH(>hjxx0by7_#oCZrvw@EbAdlvFPH-mFz_VNK1VAL9K{v*rCw)^BZKVsHZE>>TwryEh'
    'aFw6aN&~;9Y1{!9%@MARZ{O$9Nx>~v_LIWaujf`@b#jU^u_p9-'
    'pdp6@AAKy#=oXb*o8wBuev4_obk9FtDQvj|*4h@Ib{_VTifwH2&8Wj;c^6`33iB3B{3vi3x`N$>=3BOmr-'
    '@>)G3kHcvstMFnY5jg{H6K)X5y^>9U)!V8XFVaQ9_9u(qt-'
    '5|HUDxZbi(8g9q)50{5!|jml!Y)$g4O0d1c6|_?$5JV6TGiK!d;B`+L0Bp`Mw;rq2BepYXU)J%xGVE$**HlVs6y{1iVtz9)b0DCYgdG`H=F?62M|Vnz@e^TW'
    '0I_N{UQ1b8)^bCF<wR!3NF`m6Bv$ImypxpUK;Z@X}x9@nYOsOX;qP9i-tS(Cb9;X;@`NA-=E`FoqSXe*i8pp{P~!f;vkt(S@Zc3hk1JC!w+E<T_tG>e6oq_-'
    'Jc@G$xo{&E9icq=UbWM%v;N<sS}`GWgHFK_cP(mxu3nZ30*<sM5K&o;Er0}Yu7v@qAc2zAwb&p;wl@)PCptWNqJwnw;Da*}U$gZ@D<7RkN+v*D4Ipm=}RfW>'
    'kh-wVjgOd;HZ>Exor%c%IcPsMyD+$8Sp-jOc`zGkN{R0%aNTn##eJfFaHU)iE4-oWaC<9AZX4sbAfBAn>kH>LNiI_Czd<u{+c=Hrb3-;I7c-'
    'g_C3|Lo2O(6YCF>7O=>hgercV<O6jO89S)iJxa^%fIL*`6Pifts2}M3vgnaW&9Mm>1FIv(iLmFM>{{Q-fDDfV!nadrD%#)O79<qr#=15-'
    'hVIrSX?%Sd99ckDQfg7?i~#)QNC|QFLxnCt)uBqMyc(d!POSXwMVDxb%%Xg+47)XtK^i`Zko!@{@;r1Z0%SDfC8{m_>fS@kRe4VLu8&(k|{%?lDPk?*#ACW2'
    'kUgb&+5g8w+fFod*AKkyY(GJAv_V{0h?Jb-?zfHl9?6<tHj4Ya#sg(56^GeMb>Z^PauQzQwCA@-35(|eJ`OL{|Oh^8)ccuJvhmTz*Cn6EkX>Y@5ba;-'
    'rU~c6r4jQctQMZp*tLtawxFr7Nub#NXf5^IjyUjNDb|(TmcrQ3Sg&_6G(7+UMBD?caavEjIj^s?X^Zx3iNg$no*?9$jFcgoc&12ZKzxWdkImgO0mp1F=qM7^'
    '+hn^V08xQeDJn0+gDUOol`Y{{1>X8ebCCWY30`qX3;jWQt3Zjx)vIxV=dB)&^r{Fq;f=%X6DuLw%2;73MKzxIdWT6|BBsI)4Hzo`zM=p)m3E2gn_;=L#)|8`'
    'E5_x)c1%?-bOW3je`+*B%O`>MD+;o9zl=7^JjZZ-'
    'c>U)o(9`WJLy}Ti|WNkOL)fsR?1d*9h0kmr#W}rB$W6n>nHEl;8f2>Yz_PqiW{yd8tjty7wlezJSW2?Yn{sX)blS_1hcmvH-hGQ4jeieoOsGwFRzdapdAGNF'
    '_b8G;b(9V5SRG%^YmP9x3>zV6%g>CK}#=3d)2w=r0Mc;I+~^RS2S3zc8-'
    'g<!^Z$FiC_PEyD*(NTDtIJob|+e=wnW>!cCc{Ts~c9pGo%OyTAq?!{2+S$u6#g%)r?8sZ(F#65F6Gc3Tdz!+qCiO{mP)G3G&_Ktgxq-toK3@+oQv0=MdymZ!'
    '>*E%0mp-t}LmA1+PP`FQ=wMNMNPp)Oe9*#?>8T3$_4)g#XtvAULv+M)THAoM?hv+!lt`Dse%V=gadoDb!5Wwp5!^TQdeoQMbR2G@0_-'
    'S|1wn}WFAr2?W{ughO8tMJU`(n-'
    'vDyr(Vj8cN#Ci|fQvbaxQ(6ImNB8=b3a@9LdEzXrgX+oqbkeh_JGj?$_`h)fSKNA%1I)uO{I6Aq2Os<sfgWWqkota`2<mul3Qk&c_=vcdw8xw25|i+TOlY2W'
    '_qd494`@LKuEQTz?krDf*+M+9THRA4YZdHv0^QAzFrFkn5cz#!(Z4iwE#P<o8^57+7~Dp!gy%YJ{o>UOw!UB)G_nVzoh^V`=}CgF7|6*jNRW+{vn$zOyMI4K'
    '0td<T)X<#OIS*R})XwF~WMzLg9hy@FaOy>5P_N-j&Dkj*m4djp<!=}r$*i9>10A^x)wgM#8QP}?Ba%>6p%vCY75>QUwtZHwidXM7))8-'
    'RR({rRyy0@|0wd0%iKD8|<Jl`}dkFsG`Pu+0X+o_%Du|2=*ZDe*kMM)r{}R4P&(CKd~zP*n>sb1cILD~GWE6d2?9$Y1tqcVw6gwKp~v15e}i?8JQcOT}G<e#'
    '5&TUaM(CSjM548*mE!ORGg^B2KN+>5M9^!YqQ2!PqZOe0u6g4-'
    'jo%dhq?|CJiT_L+|OnV7OYs+<blTIg|D3J{ZP7WLDhboZ9ZB$usT5lQOR6b{Cs3BxjREV9ER;l6=;|2Nqv)sP^fubw;`*jBj%e#>t>@i;j2T3emY8@HKQS8j'
    '47jMfZ)Og8;(i`Ftw;h9dgZaFXe;YW)L6>fh{DeBOkyFn3NOJL<SQR_Yz<akFObd(gtYhFvM0HjC0@eoDRxps+p-'
    'C{YgnXamupKNI@1|G(&e>fq9rrTROyZ7j_FDEE_F3JQN_c<2Ps)FSG^6SfTnxpQlB_ps9=cU>;bU|odQzO~)x#+JudH<4KPZd;hSnlCDTG*&l-'
    '=}n#=C@dI^^T>^F8T>vy*rTqM8}9KXD0>y^>{Lhkmlp7CWV^{aL{%p-'
    'eIn+FJGeVFg5Bq2%d**2XcS!5+j{hTM`!mkZM$cD+cLn69>xj_g(%O`L%e!2EM=X<&`qQr+mjGmJRLHITF~zvr)BU`eyC7PP+W~ZkxA~x)EB!6E6ihUExLyv'
    'FQ9``^H`Y>=h^V81Je=eR(ZLVV)HKK7GzWEvro_q2ng=y!sNPt4T<#?tGGEnseWY9TY9zR#8AP!U@qq?9-A&^+6!82O>w_p%}-'
    'RL@%WmhP)+9o63hJ7P0G$zy-X+``o|os_+~mhlnawA?)lS~j`!f6h_l9rRj*r(6jI+)C0e)gwaY8AW0BLbqiAt=5}jxpdMVe-'
    'aRu8pe{u02^=>S!i_hP#ku&Fh=|Tacg|qnha-U16Seqc+d+O?Ua@&3?b18{R#WH_O@wO);@9=(##~M+tQzHCH&-=iW9vZ{ldK7>%bDSB)BM%rsv-1Lm-MJJ7'
    'Z801AtN?%9iOg@r^7q1|W~SGQ_5FEzeOqE5!2<4TXC1EOXg4t%M)l6SAdW8w*ibI<3rxdJ?s9gN>AczRT#nS)-iMJ?^Dng1R3M#^5z-lh7HWoD{q-'
    'A&m$m8pFqHzDJis39V&GxZxqs#L?zR*R=!3@XJSdxuaD4xpj*@kIh^9jvbw!}8i@`$+e}$bn1^oE`08mQ<1T6pn00;m803iU{|MeRG|NsC0|Ns9A02BapVRC'
    '0>bS`dic>w?r06<9q000000BR`!000000G*uYR%$_zg_*ze0E#)G7{Gu5#hef$peUj!qJn}bis+wN^Az*K^X=`O7ntiJozq=iRb8jMYFG9BpR>!;OyNHQzk&'
    'Y>KUcn=TH(ZCc<m<^jtqvY%~rcr{wOtD&lO!ymK)y{u7AIlKP&tmUzwkaMJ6JH|2_Es{r8{GfrUVCpb*FeegX%9&%jrpgTJG|1m7!x%RnPA9tZ`t0&9WwK#n'
    'Ubfsa5Z?Qa4lTGjcx^u6~$DsUcX2D)gm0EOkiVxT{e4rBvwf$6{#?IVF?;3Uuv1o@nz&t~8>kf8m2U<0}rjF{ohbznCzgf=DvhxAwvobkR7g%mub1GT_h;0l'
    'VXfmgU44@BW_GSD69gGQU4^Zd=i(R5%bFbu{0z%gUB8Mnsu4>;O}lZC)7pXI;~R7Uwd!<{-'
    'ZeFu7w?FwE}jDErhWw>kxitslXXz||U_bs1?fd_bN!Dk4%Z@iuZk7%q1o+9vYffkzZvCjC<&>3Xp<-'
    'i_S?rF2f=q1{ZAnhGIgwRr$K1+PhAVr%tyHK5DlrVZ+rPmu=G@-'
    'JOZol9(18?2*zu_~47K4nk6nKG$5Z@}$okOk+nt$QfNgxlehs<#&V^`=i2tPgOug32bv+)dVjgjYdj5c4<M-Ux`X|>Fl30lSjWA1lK|9xcHMAA-'
    'Lt~2u;@IQpqQ?`mWTGafFz-uQXpTlu4tutt<hc*}JU<dy4(8@B3=5-35zdhq~%yo@^Yk?TtkHbd++dB*lAZ-xJ3HW}2%O*2$$Gs-nD$}}z26D(MKF65T9xR}'
    'YMt<Qej8@{bx#6$Hw>TEt0nJ0(&NN(nTCaz+Eiy`(_chzcI9$x3xkvbaM4u@%AltZPcA`kS35AMnaRvK*g#S-'
    'y$QouDbplCuxt3tO0c8Gy>MGpjp|ZiXFz*Y2D*d0}VvX^x(QzE<f_@$*ng0iVo!dhCX??=TWxh|*_7f_j=rO_ld9*!*&inb+2fbA|8iw}*ck^h!n>jv%a?;Q'
    'E7US&GW8Hd_^&awV1?x#O(hKAEVs%T%*}*ebf{x@Z(RYb4!+z2pXf;f`S^8WsMhpuoG3pt!(oL&l=p?)@ywWm*F2<0t6K%z?*f#Uh!M$#H$|KD_{nlx%6()u'
    '@7HKih_ba|t8KV~}HSQepdz{w~SAQ+<JsQ5@?k;j%^O}WXH=|EOHx55J^l}8xYjE%e)ob3n81D-`7HAnlrafBa(8?HkKSaw-'
    '`cEKp8wrZ&G6=7?wCIM`7452AIfdd7oLs^81afuJZ<O(6m*Z&X0Zuk)vCnvE&;0}3enIa53L#{!qxTq+B^aj*UEHDTFWRRWeFa`x+&$oP1L`H-'
    '_n`KUyhmPv*J(RKTdlcS=;x48>%atde2Y~bLh%3{Wf?`5Tw*>(cq)FltEcb_eV)Tpo4()ZBFQLP8Kw7>pUpO7uOQzscbbg22klKbctJB6C~7rcX6_%M*=Eir'
    'JX7n;M9G{;6Me{Bw`^NbDbk}v?<O-d1s7|4exR)*ByT`v7G8dk=!UizysGrv$9i_?c~1L;xt-;`0F`TWv4wWE*4N>=n-'
    '&eY8fUhqX>|aFtfeYqDP{N`g4U*=lS@Xf(0`Qi&X9EzDUQ(76#R@L;~wv2Xm6qy?RiH~IicmI=Qs?Hbw;{?tA3utHNG8T3mxeF9iCcf;Q~GyNU+Jb3OruYzQ'
    'jl)Q2eG>-'
    'u#a+Pb=_#?Ok>eySPW<N5*bj=TWX_(e@%E?IPg@tuoNn{C!$WnvHpW4bx@{`HGBt0*`x)vx)?DMtk5}nb#H8R$}C~SKu{9(9;rPZF+(BCf_!(xk;~0!*KS?t'
    'ZJv;qF;zl*?$2}R^jlHkp{4req_GG4m8`VmVXI8(zI$Z$}yu~(`JV@3+PWgB)OM_?h88Ghtn&M@a^?v9p0DumZh&&=o{JwZ3QbxIfxFjUZ=L<{2J<U+H9ePZ'
    'dwj_ev;T+i|_Naxr1-n(hU8}T;JyEC-XTBr3|l6Xbr(f9ZQd(7d;J<at?kskbd8*SQNX^_}AQ>rdJ5*Wm#HD&as3L*Uq4?{k+T=+EdeTJx`kjM$OVMjZ|8RQ'
    'jGP&Y&Gna<ou-'
    '686oIrr^(+4BRxPh2T$Kn9;5$0yac&d;PV>px3TRZ?rMKd+xz)MXD`^!HFwV8YYHkI*meV51hLtDBx*6nD0F2(3GWc+@Q^^34rZ;+_`|lcWk#*q1G&K(wrLY'
    'V14*=ajD*{K_Q9WafH`J1!%J7*nawY_D)ThHdPV5*>KtK)o)~oxo@QvF)%%IIQ#?@-'
    'Ki!)9Pq;6ljal^a4R3kcPVzfKzY^T^dq;ml*YohO!;IIMrxN@QS&yCQ;fE{Z-'
    'hbtLzG9&R@F>6M+|TehG6m`T43!jPWZ|wxs~s$`51EJlX=9l_6?({*lb`VH-Fp_kXY9>Z(4Z_e2In#U<Pqm-uNCLfK9YQ_U3-RW_MgjW@_`xoVbpHM-lO*lT'
    'xyr^<-L>EkF=dc_5tL}a#b3ehv!+SZ!y!~(36#CdA{=e8b*WfaK3>Q7xqTF_%+J>ym@VvPdmeFkH22;vfH#6M)PO%-'
    'sQJEnmK#0@*+l{{KecA;A0s+3+QDKZj#&;Poi~TzvG);729`&=dFNLcZ@O34BpYBlk0V~@&iZ9j27eb-Os-~jXvvfg^@B)4P)Jl*g%}C@;-'
    'mO&!({QBj#Kl`8zb?*hGVN6J95i$glls44E7B+JGl{yYJrfvRr+)Czs@Td8AJQc~0SUgWmi6exOegJszWn9rQ4Y<gd*AHd-'
    'v9?+PQy`^YkSh2HWbf1&wgU%tVocAT7d1g+?O%=|tx({H~<{-R>18gzf*@eEmOaJa-+14tr2WdQz$Y>Tgqxl5l-'
    'xSQwsl+QTDn0rtz!i9FBS>)P)iyI^wfY&O0&ggk+k9P@q?&<%A-V^X5e|O$dR|adEg~wC$AEnQMWslHf9_!Q0EqM*PK&C$0m+0|@-'
    'd^n~g|MYfILtAZEw0EP9QMvv=E?;6kyjyKOn%%BznY9vq1QN?kZ)AM7Gxuv=qH1=4r%vb5AW0a(>|XGuLW0(a^-z}fH4|g$sW-'
    'E9QvGrlL}JDz3=MDR`hp^HH~3;TFn$+J<)H^TGQ%zj~y-'
    '8D(bYohsp?Tvds7va>^rqfYvE9u6^tc?T#@EO?bIP%d&(O<lDD@rB(Qc_X~RK`TC&eBD7A>L7KTvBSn-p=jb#;`#0V<X&I*X0b1)s!-_GY(7lG=E984-+-'
    'GJ$YtRe4=b6)zqsI&*N4?J9p|KS77NPY9g)}@&!tEG(sv?=Z!3(6?hI4t(88mRfyW*lGccsZ6$MMT>(98P-'
    'HhTWgv+v{XB;zKy7O@AXeJuiA`78U(L<dwB9d``E@hOz!+{<~!H|RYJZ*x$;<Z7DN9&<8epZuH=ei`!^-'
    'B+<xd1d$Twn2*{uOfM*&(mv!YZ=>V!n<tJkw+4_BaVg_{n8FSNdIRvy5~qL29;Una)r^gv-C0M1X6r4-'
    '(Sq&Jw4j=RAi8b_fvT8_6k00zhD539@x{Hru7M<?l3dE$lT?~w}aQcpMrO;$IwR}en#jmk54<`9QRAiUd1u}GjiNOtB31D(EET+k*m+N?C`Ffgi_J_?F`SvH'
    'TDz16675`(C^sZ!x_9QPB=#X16r*><<>FGD{|-'
    'I;u|VKu9eYl%KqM|V~qr(YL(FLFYln!dy&wB1zOIaiyNMzCpdrLGlX4)u#zqN++#@8f?5~)h%)**a?0b#A>9vj=NNGTo;tWTK)+?^576op%h0~OfW0jH$$Dq'
    'r73mjgF$cdw06u8h&Alnw%%Go3?<$k%zlD4eWPhOL9W+{eSL+70lVsc_uI6c1<<0?jh8V5pNb|(;T+L6~2;=9`NQ@D#;aSgH&X|EPpfbAq;_jd|_hhd*PTvq'
    '5$ZLONj1lN}+ar3%GRnLZSqi6-y_9X^CG<DW9IV<K`hea8T-'
    '@8UC|FZv+8jb*8{JKrwqolYxZC1d&K}(e6t5U*n`<vda(2zpiFfNyWZQ#V>G}rUjxrl*>?`HxHVy3+T6OamrpGnX2|wF`?x*9ex}$>{9A&*8o*9RFwy$z(EN'
    'Rr9&nX&+nXe)(me97=;(PANvZ~(GCg60zYxx42TY{^kv9UMrPBnPR(Q+4lAB+QydKJo{bIs3<ZMx>QM?2G?eVQY$aZlXewjUX@zo;Fx&g;tF%O%pvPx$bvpl'
    'EO(Eh^GoGH!8?#3N8Ug|{=fd8EHEp*#3`FpXcnb=nK-'
    'VLmkbqga0)+6VS)ADR1CI8Y=qOTU)2t@TqY=sf(*c`tk7iQY8g_zB$u@5o8!RhGQ&SSSboT9M?d#n5FO`oql2FYh~k{!89b?@jBO_6tbyw{Pwv=Olc-dj%6B'
    'u!P2LX*q~a7Wmz3<nq(&P8vyyNE`BOW|*%T_*i0I50R+bdu@XEJwD|phIlm@x58+$v`+ebqoptVK&xo#hWpz_y}tN;%Ip-'
    '4lN4D=tBx?{peWDhj^72w$wRfy-'
    '8JvdikrHPIw`V=!jE>SGb~!(?X=_GA+87sJVx4nbaMeE#Rc=s$O|+#c)v!M%SIoxM#(OOL*5xhmFFR!=Lfwjwy)7Lh+fvAD?H*9u7uFEX#2zaIouz@*#`QO*'
    'B|6=pV#P9uLas~WocEeu6pI{gR2kTwZ=}+Z<q0}xzopIlOBEckHTJ^|00`;k?3n%&;-|nHft3>V$Nq6|I(iHG&8shhw=>?+>OKCwdZOF-'
    '3sB1GVU^M^fVRF%qmx#ydTka1{wbFhcfycW2`amDIQ*>^#pw1bLG_d(hYRA12*l6%ow>7{?>Fvu;yKGlDSeebOc|0_H;MU>>YeJ`S!_3k4C1`_D+Nf>RsBE-'
    '`s(O7g$k0ma6!%#FZxf3-(l{>mKxX2(>}uDixz!Gmg%ie!hj~h0y)EvD6?MS+!U3LXQ-DB<<r}GRB;v#T6rhbwBZnozD3_?mb7oSB-DdW{`J1vBEF3dsmE~H'
    'PFbjy*BydosLmc(9zo6fXg5{Tkw861f8VO$bQCX!}B^bFFb$Lk$T1U+~Zh85#=yis$hL7cojn3=GTDj;l-=e1Uk8ge$4puDeX6DJBXfoki9^=wyh!N-'
    '9|Gz2Nyf`*9*v_nD!U$FPX0=xRs~90EHm)BabbEekSeNH2JQzZOV9L51Rg^-53<YNVR6vQ2Rv)S-'
    '<?myu(wP`~9{V;cq)=`P8fKJ<r1@Hu26gbO>ePH}aX^u%iLv5?Qn;^uNe#Di89E48ovZpr_b4!k;v<?NvtJ-'
    '!%NM_^GJ#{hAT<wC<qaUF5CN@7o!QVfd`VyQ1K4qtZu?>VBYg$*(?oEi=w4x{Ml?uX}Zo2mj2qE&IAtjQEI5Q^wZ5nZG0FUFztkkN)RyJHhWgboC6~UVA)>{'
    'hRQ1%liO5-&q5~w!<Ih`v&eV;dasZ_%F1CY4*TF1G{Xa_X#BUHvSMpi{Jbeu)1kuk9FsY9?{z>EjJzeXSnyx^$c^=FoL{bjDHR*)ZTLdFHtmE;+vk}W%{i$-'
    'lTVheQ14pCmXSL3)YR|unX@a@`M}oN*U!8ikLPY{7gULv&sSp`CLP)Q><aq8YnZ*%4u~tQd5kOWvp3vNN`ON<tto{Bl#S<sTt3cR-'
    '#aNf**Mb@<Jxj*^F_U0d)K19K#S}?!eP1w7aprU9JkN?(;J?1LZBWFos^kUZI6NX-'
    '&z|>dGE&2O}>zCfM~p_~Y5xf!2+8vmCEJdPixi?9DVXz4Pmb`EPqo+k&HnGhgra=5pBBHKTPn)=aRP<c+g*TEZRd@Q|Jt&S19eY30%A2D2cnLXq#KcXCBlQM'
    'B`o^>*0OkI`S0_Ii3&;ZITHH~OmDj<kvejp}LzZNb-'
    '|EpFV;bi~;9D4JNe*LC6ef0HNmfRQTbpo+GHuf~wH3tjDdJ(8!b82#B+q)bwaK6S@g+qN_z|8>VqNo=DFT7R>-'
    'iZ*i4>Vd+V{p%<k$(Ek1&8p+&N91TQaxWY<7-I-'
    '3+g#sd9y;Mbb1igSsOXWA+gs*W+1Msj&x|tU?7JKqWzb%#i1fj>Bh2^CPtmT?v3}<Z!ce@1|2QLc+XEf*UU>sYRirIKPpJO@pU-fA;ayZKO2Jxh7^jn0yXws'
    '5-#(_DbOl=5-q}L-'
    'Ez`zrggqyn_ts9*Ya3c%yfc1_zj>addE`|b(_!0*L+z2dY}0zmRuwW15a(*a2%@5QW#kmm{i2Dy?ds9{VwUTl&TjV@741iVzs&2pH7_5z2fE|l8;{|3)VSBX'
    'QP)m%F<?CV$f&I%prms=LOO%cP$pv;x-'
    'az5p7g|>32Z|Mkm8mnu4|v@fcp=xmea`h>G<f{5k<%zU(??GgP+NFBgHYU4xxGZoQiCOWt}0vX6Tcins51G_eM&Dp#4c2@-'
    '8v!ox2DXAz$6Jy*FBxMMGPTMU>&)^h*1NTxsuu(r@!$ymfB0VhSCM(6i;th1M5&SVGqpo$b(?{(-'
    ')(oiUAgM^AEBnR0mzpMK&Jd_RT{J)8Z`u)Y~bTW}^N>QzFx)f!ZkiCRF1kM@)9x&KDb2%4)Kag}AX7;}$xB4a#B@pP|iDdaWDt5Do553}V}T`@!nN;6y=VT>'
    'bYc+V?B23lL_zXPf2NS>hGoAvkNnQigj#kDi%Wt)t&g(UJ5)|@{+<8uV6XU0?CjQBmEpH;_)qi~n>DzCg<kybfub^|J#(AFqj{N<sm49XW$C{Ot2HSNYT-'
    'eh(*v9})X9@)cDd=jL68VxK!y@XzK*jtaYC^6(dgK`RsOa6-}h2(sAr#>;lzGV+EZja~Jxu#ibx5xDzQD~KE7jraRHl{Gf`zU<v7`szOO0klV-YayfZ2q+ot'
    'r@h?V8l^qKB9N!Tf2=+q_KKsMW1<{arMI)s%PhQf1PoeK`*DwgI1gZoMfF*nYR||NIb@P^8fPoQ`hZ@h_?+_Cal2MnezxmqYfL;NIM^Y>C9`2nbh;8Z2OzhF'
    '6p~s93;a%?M;ds$6RgGXPobl7SGPmrr`I&lw{MYJct-0IDtyd_OysbCtY)}g6uV38siC_=8-'
    '85=S#e+g5!=EA9H1de1b=#uFA1zX+P$ip)x=6=+3y8_S#fohUCfD(9K^>%RN`LhG-XBGeROXL0N!DV_vF#>E%zElc3k2qHQ+?kJ85?qs}?Q)klvlV@XqtFC='
    '2gxkF(~s^wAkRh~-H{@(}Mi&-Bnbfo<$$QZKO9_}a6PLEgoW3G&QH92q<$|1CsSNeharQ@O=WE5WC<xFbcSl}|PluuA*FzeiiGS72<idEzB<XJj*-'
    'cGoG9BnNlf&8#AdcJo?Ytd`Qq~{}Ql%fdLMSkx)S2SWPQ|9Wx@$!eCSLKt=7*kR33Y5Rm+@ZBsa21O(Eu(OMZr^ptI9QbyS|=6f3@}E-'
    '`=oY8t!gLeQmg)ky{d>upE3IRVU%JFdS6~O3)W5DHBN)j`E>lZ%V=*<+4V{rwzsQFg;zLT^|O0{4Rxa3Yh)^7U0T7fjXQQa*P!QX%g^}{`cy4fuQ3Z{Ftxut'
    '!exs&NpW4(K=Rx5^xknjYn$8lPVzwOZ)aVmoO#pAt=ZCUw2C}JiuI)aFsm2N4ouPW)m2p=jxR=d7gp7BEy#yg#Up0t()r0(EbZ3F>WSmMcYb|3YdQ(Hujp3Z'
    'a@)Bw#qih80;QROJLYeZv1GFoo}IMw$!TZ%<r@g09&jy3*x8x|V>QRlu5?4?+@5U++Mo7<<!k8~`?l{L^$z>xnNS9|hihBtH3cWT#&v7<7dD--'
    '7V`394}Z{gdkN+!>PSLQqV_`7)@*Y(;duKNN~6$Mw728xg>R%;W|of~mn-kMZ@e{%CgS!f4xAPGWez4Sf7lVA;-'
    'HkH*m3l>j`Xk4xi!|H)i~@rjuFQgL!SG#5%ZgW9)n`9H8>w-PTS0B*chEWjyhL@&au~Bu`tS5U#@!_gzExS7Pubcj;eZI8E2evm5WRxOP-'
    'l3*_#udq72dnJZ(Yk#I*`tu68WAx~t^ap-'
    'dlPMOAuAvv<yEszTxwImNLszPQ(spmWD9;~J}6sW}Q!#_!gZRmyDK+7BIuhg;~+BF(5*P(>M=UJs|yXVduj9X%(FK`gtLNBen?XW-'
    'fNrns+|E6cbYwrSPG&0}LVC{$^?L_6iHlux>K&E=5m(HgFA==FX+jy!{o4j<7#-nEY}^jI-Uw#OJ-'
    'e103(QS5hVynMzLGm0VgSLfHfb@=1jpJDqLpRVkiF}5OfE9=UlUG((p9P6uN^O)_lA5E@$_3tr8+jI`@$vE(@@h_of*Yr~^;mtdn#!10n6p6Hw?(n3xk!9R<'
    'RZa6H@8H-g^$4F0R}Zban&rWM&zvKZtmkyX6;AW6etLv*$jIoDJ)1?ZQ>q<M9R7^9gxU>oz1Ozc13$7-'
    'p$tjq)KBQEEZh~J2|qgt*Man*8PzRp@u_@oh^tq~-_QFESAHBNm5_H7datg)sM@Pq=1-'
    'C26Fu~#Y9DA=o66;B4{o`_DDJ2{ZtA~`B786bFD0XIiZ||{5%K)>I)m5CJjb0ATd_=9wN@CT_g`L0bxB=ZPdV>;#}(BFw75U&>d%7f3YK`?8!vgq7K^TOT||'
    'd1*D)3SRDNO)2j2B|9sfT$UvcK$a>*4hLUiwqMlX4ls=8K4J4NzerB0RE)RTGZ%$u;Wx^bDLabnfN)J;z)!Mt-7%9g99Y{!)lYpyI%?kW!-'
    'VOJX$;Ch;?%F#?gMYws%*`PyuZy0-gqUD&^iy}Il_UavX)u#NGBYF<es_opN(B%aDciCE%uQ;REzHt*F=bMfi&)`tG<O}-`8RIT-'
    'W5*5Wa#Vvb1P#?`yu!(y5w<I@h}*^&MzP%vBRDxYQ(jeB?5ZQg6-MfYg0Rv$Jp4LV%+vb>o`kADLHonmmm)25w$lz*`m~*W?Zu8|`)97fJm8yhrb+KtF-'
    'Gb@KQTt?;@cG0LiVt7@G-=^cX`d(_j4uc56-n|t`;1z%Om=7Zs^6R<)$;15p*<ebaL3WyZx>;S!RAzdzZ9#P~iQ>Xr=UZ?Pz-$?w9NztBU-Me#49%Lz@lP9M'
    'zzpiVsz8sRCL%!7cKAc(#j1EmX~ubXB*I$q##&C0FOxU1|N~T8tgm$$Qt%by<^jbgDSu*}2vjRJAsW`m$ru5hNe;DthLuX$RbF+9Fg7`CyE61g`HK;ZMQiFd'
    'BJrEu!kn9*yaYxz4oassdFpDF)g^4pkvWTtg{rLAA?MP)+z*zIJtdAJ_68QC_!pn^!*#!doBd6*B@i;8j9%FoSl7yr*2cO$w>{y~Zf6P!0V6V-'
    '6U7szWhlq+*E{w_e|loK>m)%V3^5lct<{*;%`iSMFm+aC!T7s^L-<-~gPpps`|)K^br5w^gUP>pID@ZBum`suHWaZY1uC4f!_>ukI;FO+u}OSXW%{Ib-'
    '>h^L;cDlCqzWcCBE{e!wC<ADy{9b4IMhYn%C8_xd*I%*`6yk9c2J9IwofqKIYJC-$R@3|H=;e(Rj-'
    '8a)5)R6F)HRmmkBO>4Vqno`az4KnJfap_L3hn%q%Ui9cE?b8_B0=xu~dd2pcbDOHYk_+1G+N=IHhIGQ6XYA<NdE-'
    '1fSjHj;UBNa4Z};9EKU{tNi~M)a`Fy!Da*$v9&Zny0c?xM__Jve)GR7?2pt)UIsd8_`_`j-TPF*uN<*NQ^*B;AHkycu?6;jsW+AP%)<c-'
    '{_5^E0W4qZ_`>8cFv^QvOgn%jXr%6C72>vw*uM*hdEV~ttp!un#iyrydl?(7jyu{PC=C46zGK&v@dSswbNK)>UdB;)3++hgQ7H$p96CWIDMX{@Yp-'
    'Y9t!t;{%|wqS%_k@lf+^hstiV^3D6ByOFr*>J62!Dv?8`Oq6j`%y-2)4tyoBg%CNExg3eHjN}6F-q1|NXh|((Y-2{B9{Hg`AB62BCZ_h@Ckx-'
    'EIDsnr{((52qRS-sS2ymyyRWy(dUy1@-'
    '!w~5f`#oc3`jV8NP>%DJ|J2d*x3jKo%H9INBEDETi93<X0A0NZzhv(xx#uVU}I?$oi1wzzDTakSDMBs=dDB>LlMo&Kmc+E~Si)zR<#;YnhZ0SVLcm<y86m;J'
    'oetEvIPt4EI52F61vM)=R>b&bugb?We_+>pT^gUph;2OS@z9C**S8nYV~NMfql`vbr{QGLB@bIE+JIPr)MESEet|oirT&x)yW56;-NVP-NSJ-'
    'o9~?ZD*2ST(6}yV#_;L!#O6^XdgHd&>4pbN1Lj*=(UDaM>OXuH+dM_@VrWoZ^!$Oj%<Y>={%A=Fhvnr`mTE?e(@=emVKmONA<@(u`uQnI#skDH9nvknpam?='
    '|q+yx>K*HRpU1WXN9heg1x(vKS>{D&eG0U>a0n>E4=QF?x>n=+c<63ClKac4K8%_6{>rFisEpn=km#`jPf>rK8IG)Fu$j{Di8Vu-Mu(=`839Rj3f;sh^ka<I'
    'miCDn#?ldHFV!xr}~Et9JmHP?D#3;dNb{o*RJ-d7}H%hBH7FDOP@!HGunqSPMzpDwok2m=MQqM8X+BbolVxc7<pJ5$fkX~j}|)3v_-'
    'p!k%3cJ=Zra%a|oBWaPs47tw-132{jz^>hV|Csq-Mhpo6Z88nyo)wU6C)-'
    'lEIr;eLIZWzw_0j7F6eP<@N4feJoFa>bP=N6T|wy<VNW`9Dn`^A~bHGVYjT*l6I6&p_N5JHGsP4(G&`IKni#jA``HR;Ti|UBz95Uu9PMY*X8gp;c#FMSB;HG'
    'Piw3MfDYmU>eAM>vf|W>gRCK=~`NyrBn7&&;OUbZJi$Yc8!c?A<yRw`o2Rq)ApiNjd=>C4meWgS;*us_d9%+O?9@4IaFPF0*9*W8-'
    '&6t?YfQUd}0I2Ln~u)?ix4M+o`%BWvuJfUZZ@auul<m8)3UJ`qW~C9Y&w>7_w7U$ERtr<T^B+zxu1XRsHnFziI{H7~`(r{9!hAdZEh|l;^hP1Lxisy??Gb!d'
    'CrV-SME-zgHwuPUgfX+7#=G^Qx<ud--g7RTbK)yh76+Y14kf7<2GrpF`fVPG7aqZURoOpk85S&+Hc{&(m^_PSrf}Wuit**F8#tHV4jJ<$U^az;=EM#UT8}ky'
    'kaplJU+ynKFbr`xf@8tz%@72dg!6%zKS8vWxWE<oX>No3KCn=f^8Mb7EX!p4K<UXG5+rxZqmB9`cj36jwe&_5pW~Ms$T3&AJv>Qfg)V^_hwWwsZye6Hpy>CC'
    'Hv}^0qU_IuDqKmnf~axISP6;l=x}p6Y%h)&)nTtM-'
    '(27IDs%F+nJu+E<g`d+oX&NiheP^Uh|e0(ce;9626)#7cghSqQQ&yz^aXOwwrAAw6E5SKZ~8PS#Dr!MssZ`Oi89F$I-'
    'NXKDLg38dP}8nk4w%Kbe$dr@$8&N{MGT{F1u)k}4qDf_+4-'
    'aCiv8{XQFR~5rIeY#xh)pVuMp>6M;@j8uSYep4+DK~#&T=#=NofmpAGW_n`x)9bh+6ek2iq?iGbT^C{v|VFgHrg*Nu;5y}Q)dzsb5`uZsnYq%Cu?S%F`IF9-'
    'l}(nhSBM9$B+$Y3Y5`PY@BtKLctjF1!L&0GJsuJPu@AQHP_s#vaV$x^M%<O^C_egX6FD(tFCzMbzUv*D*0Qu+HmdNw0F%dpWJORW{2^fA2dC~So1!im^Ch+b'
    'k6RZv0r>X^v2ZYyr(GFRkXj==`+ux#(t_kDLn!weON{si>Mp#9fsl*?@{APs)ee;gSgU3xE=5KvY{=Xh5d0Q&55z|yepZOjY8?n(y8lKb&6oyUTem6SjxKUG'
    '^H>uRX<!YCs&SZ!@T4X-TH)p_Agc06@7+EXzZy^o9a3LvUhRB=r_pn;JNPdY2RtrGi#N}ImWwm6~qX-'
    'k`F%weO0?I7;%|!)sE_^l{@R?w<7mL?^4swMMs&(pd&$9aLUnv@=!O<emr_*iaHBA;hIEIiTgZG-'
    '7CBPdYm(ybLEIClmBWVl>=3*GUSt2Rj=0D&hfNeBmTsldt)}$e^1GtW0gZ!lgQ@6uGjkYP84;8(>kMedT&)dP#tOR90?vW#;hwhhaH`(0{G0BSQ<^P*wP!W_'
    '4)A(eA~Wnn3s2WD>zFYaZXxw<u$aYS(#>DRC6nIHs})r8B6x;dX_oI^+hCJGrB8etLz%IV}AG3PuWjp6PH{u8$xG0UTft0=NwUV8c*(aCU?cyX&UXsU7<DY>'
    'POYQ>)SnJMvRVDyy`6eJ8{!Qiw8d6y9OnpSbBm6(A&AsTI@kXdHyBm?G#&mnno9#v|a5f&)^Wsd$#eMb4ogiq$eO@&5Y9PuVPl!8Y8X_)&BSH?3ZeDIz1c8$'
    '&_e$0jJ~epz8i^C_EVZI&odVhGU;6<CYrb$)~Q~jTg!zT5t{hgZE0E{+w_H;sctUa=z}<Uhb-'
    '^+J&a*w9>F6rUPe&V$SSoCmys<HsaGDW3C{V2e<1L^Twx%2GHKKvFkV5hJ3PO$ma!AgOsxUtLEz1KIfIIq_VE)RHQZMT}|t~{ONU98D-'
    'IQ#;cj?j}?{O+fUE>to^uaK%+iC`U_uSR~G7Q)nDG^$vHQnpIM(u2s<0T>8g?$N6?E#zplK`g?uJx+*u^yqc2{o$KY?wnBWrhvOaUs;o5KQcq_C`@Lfo`vg_'
    'afPNXzwBi~G?m5*!_ZR8q-&n08fi;h%vD&g=Sil8%bOOBF-'
    'zqPTlhHI9_jp4~xExHzJmNC!&#aqh$3%|bixnRW+X}F$n%&apT^Kf>Cojmvn${Qo!hT4s5DQ28!3nTli_nl2=@t0iLbB?@M^cIGvoxkMldK>AA*vo-yTlBwS'
    '-2FSFA|ziv(x^{@Za6Al^2$--{*%vo&N_>zovhOt%>(<`G1u7DU2&`@OM7|Al^Ghf%m@**tvbei`YK!b2E8BG9SKod^>cDd+kWR~t8gt`bI|L&p2bH-'
    '9<?_*<C&TIcjD#mf3c7kBmd~f2<48??zO#g>I6>9RTDvEn>7BdTBBG0{~>pN%2g9|Zp#iDLr|^dv`@?KIPbd+4^Om@d;jRLbu4=ixP=qNI$OwEKnvsEZ&v(2'
    'h}3+FyXHKh&Op_i8%sDFplY)&-v9nL0{#C8-99Cd@`;&K*E$?|jn)~Cno+|`?%y!_ozE(-'
    '8SBtl@gG{Pp+8j#9J!M2#WF5iI%O<k$Y1uZAe1WO|FK}*=Xtfl%-'
    'GK<J9oY8^HX_Sp#CR^oNJE`uuNr#KW*bhKdBWXLwYW{9WQBjyz}Xu99&)*z5DPv%|2$}&EEf{YrChBPp9~mv3ar27lHS%tG=}^Z`lJnaIKfJI4xJAZThVKug'
    '`+j=y~S7CgK0BK$u0yn5F7Uh5q+DLbxz`{9r6%-EmACz80NBn{yqwkl1IRch8vKpew)EouSY8tl))DQw%sd)H%~}bD{qgW6HTI-'
    'Me?5e8W{@Dc83gdv)0JIjvtmZ#}LjkxVPjzf_E=^*NWf<WrInXK!-eAFH05yek|x?dPo;OAOg-Ix$waYYep8*`_Pk&Z)kw0sV2tzwn+`^J(0iy@R|fng@J3Y'
    '}_mP6_kW)E2_G5MM~ZOH$}#~$FBW>E&C?Ju1}k@Zq8{t?$hO}Qjrh1;&V9DKBs@;eC?-ok@soYn5)qSoH;-8$@c{#U|QdVj@Nwl_0*@mHXPLo8`u9iF~hj0-'
    'sSb#<!oJ<SNh+1mw7bvW6$t!ty}SV+CB85s_1W{&mTr)6o+YzTVMus0-'
    ')scSPxtgmN@7CSx5gP%c1K;R{d<39mnZpXVm!SmsiYXBdu{H?f2?FYZQ3O*#D7NxqEu`*ke^KSf6XEE}T95_5Z^X3;ZuoO9KQg000080000X090TkVgLXC|N'
    'sC0{|*2Y0ApcpWNBn!bY*icZg6=401yBGwE+MC0001VUH||90002(z1LPF+ZME`|F$0h55fp1;k^+ag!e{R0RjXFf$%py`YHN_JK~dy0*c!EoW432xohB3na'
    'k%|bH<Dr`~O|rUEMhNKcV-'
    '~|J{6f^>XptJW|^nf9q*(t8LEYpI@KPA5Qbnmsh@?IM2OY>H5pf`Qw%TZW(B+9q#S!X>F_he`^2#{r7)ehWbKPp@Yyy=q)rKdJH{<O7w3j)T-'
    'aFLc5_{s3jB$O@`v3vCxjL42K><rMmwxbgH|u`nStx_o0Q*Rwy4T*ByN-VJOrestK)!HbZxzj?jhfZwn<tE1}m=nSRFf%tUB4G@|=cp>dVFtr|D<$$n@iRHx'
    'C1h8Fe2SZGb(=TyRi+OQtF4Rwe1RN`ppMy+iLwX1#6P(`R(WxVRiUj6G*i#kFBp?Z}#7g|<bud4S}*B{iPDYc|8bfll>p=p)UsJ}b)NmhM&3010Zduq#qYF|'
    '+e=W1m>bgcG8L(lr2*WX9_xfnWA+n&|Vh|0aw*JUWJ5vx?2+SG<^jY3}S7*qY1DyLR84~1rRC8c|2Rr{&#Z%}Vf)P{&gsaekq==Tlv;#K#|sMK!N(yVbD)l+'
    'wBMP8-MX>6a=(haq(LeC%SXGEh|rdkF<*J?vVzg?-^ZuM(JBY&;GRzgX&by0I%s=BZAOs(2esqwqj-wT?JOO@?9lKNWKXx?Z%$~1<}x@$;vj_A(8P!rFs>iI'
    'eKWkNkI)tzIS`4Y9iPQ8w?Dqb~;p1%gQwNy24sl`>gdqX2tse86H2GeR^Qe|zb7SC%;ZGK_KyEWIhdM+O7QR`dOjsvZ|#ZXwiEmO%OYWKBTnb!=Q=(D^=>s)'
    'uAY6N!FSGTiCb6TktaIO(~SIe3;DucS`Q2(Cw+n`o#iOO7L?R2OWkBsZ0?mbp5=lULJeYB_*-5R;H+Mm{VE@%Y2Hg+{T?ds`-'
    'O1NSz4r~3U)&577;nmQkT3glA8C@GuePQ+aNu`debxD;nu4~Qu-WST~`3toouKM;h#)Ilz8Sg_>^M9tlwpbxGx_d=6pX>LS?tN4#jT*-'
    'hecr3luG5&;=(lQ>H>wuZtL+E+G^x?A&>XL+<OJ{dr0SW|6Jv~<SMQ>J8`kPsS54Qdw@RySK>aMyJ9esa+}~3@JD@t7dDG5xSH14*(lgttqem;~RJE>YRw{J'
    'YvdS4@U%1ho8ybry^|4f=)uR>rs(C5VXBBEwQoWheb7Q*OJ4}y8qhELQ>i0eUmQfv5D&<z6Eb8wTeZA@GJN-'
    '>*gb(%UjQX*!uP&8Xq1roC?x5PUqj6bMo8xN1ol4!;_j1+uq;WjZoe}kER(I`aRGKvIiyGy;o^Ms3U)6(Sjb)kIdZarlRMwvE%jn9gN~}{$_SEiH^{ZUZHL8'
    'BE%NC8!nOZWTJLXj1I&*)f);_7cd6f`RpR*eG9`$TQ^^|KYPBhj}x_@1@534QD`gC4D$5r~NzR#+(d-ZpT9e7OlHt1gOxtl6~M}71@(5iKOq*b-'
    '366ZBWo2tbt`Bd}Kpts^ppZYC)(s*vEO|N?PMPrdrE#4VD?hCw|uc~`k{ae;2dDS?pvM1DnYmLl?O7w0#q`6P4%va4>D>F5wnK-2-'
    '9*Ju8IZNLrRm!oRIMvg6%}h+Ki0kKrMr%nu&#9CywdGAcI@G<}`pW3ZIjx>)J-'
    'Ma(M`&%Az8|QReT~JWM%R0NR&B1(9XYkCMYGkRyXIBGCcQe=N;y}%>r~bR@5!!ezS8rJs&7qwolq~9G)^(Kr%`>J)%SCiJ)v>&nYW};R&?hCbKI;pW>wR+T2'
    '-U>Fs|R0v=&M<?)PfbvqoWC?Z~MI6Z-8+ZQRxUr>dzzCBEpXB<*j|JPoVu`<$}<S{EtxIIX&08S{2s-'
    '_&ULtEL(Ca9nq7s9ewABctTm=+$5Kx+kXo9jo3IwQ*MUjH(A&)pn-'
    'e&h@pY)pn|yU)h1<s=;qdM0?Y<%AU|~6Iydo_NIEZ>|L|!lX_CmMfB6_|3EDnRSS1jQ&{VzMtwffI`C|d(*FUqV_kPWtCnTezOQ?xbx)th&uvKPvxLgM)|kz'
    'yrF+!y!hSNQwh!sIO+D)!`cU_lu?mLO%UX@WCi~QsTE4H+2X)V+MxjD?hMAv)*4(px@6|mgYPZ)?r=CC8^(kF_)O^;fqz!#Ns;oM-Bde9(rg8Dx;9hpCz2oZ'
    'r9J^S%)`jce*QXtNDx%(dWqBvr(n^Tv+M3Gud48@se5S6e)xEl>Pql99xpno*JJEvby4Gyvuu67#Q?ja|jCUume+{bXOr`FqO)o0BNzc!zEoJ)bKtK1@`YEm'
    'LI(_Q%c^&KLQDb(kwX?5Jw$!eeN-'
    '5FW&S@;lv}Whjqi5C8sB*o6MmQn1)P@oDr9`uqRsHp>vLV%)!2&tdYM9bJZ5n}uMsZm^oYK!~wa+I&w`O)jU%qm$*?dx~&h<9lup?BmJGW_uE>!ER+SI8#yn'
    'A2h-k9E~Hr{T}{i9lctkLMwxV)%sN!=UO-)(yCRIRDuM8D8j_o@xkn(<rB)2Z59$2gX19Nu)Lh4a_8=Z#iqSZ%c3v&B2yqCS=B*-'
    'MqOpgK0yx?9~ftrb|UKGzjTV@S_j=?U9%wi7Nny}Q)zPORB0ji6U*k6PZNf3}E|y5Boa8XL(r)(n<m9Q*lPBYCD7c~h+ws(V&X537|v<*W3)RClL!Z&ZB`tA'
    'Cri>JjTzo4Zu{q-Od><$0BF>itUUuX>H(y;?r5UTkBHl<TiX&CeZst8Lm&ea-4$6{qZ!?x@$uuj%O-'
    '{cVe;8w=J}M1x9x(%c=W9Ybp8fySj)tx4!px5;JAV|TphsVmlZo8Gqr_3A`*bZ7=o^hBwyXEiEsYSEBt>(S49-'
    'hW#f)r{k?YT8h#&05|4S_6Z+YU}fjb9O<id`WX|i~L??3~Ei}bYCm`WJ3M-dDWyo=k(OL+GK0@o-'
    '=P#SMRap5_(_KdS*oZSyfBN_4J(nKGQSD8pmag!?ebsQ9ZxW+)rr~Pc`0Gs>#;JrfR>^)3zhuRpte@d`>@o;_PrDct@|+%+G0NYOrf;FWPsyRk`nK<C^+<s}'
    '>HZuCRJyJ0-04)v*?DROgJInNaI`^?urBjH%99m3*vL_!R0=zsA*yL-i=Ewr2Fqnx0(6;vG<bQhNSQ<36Ic*xv2M>)Oz2>QWn5HU8~-'
    'W}d#c>4{#gKF{0$d(gJ}RIU3@^~96L?FLIJqO~-k7Vc;+pLNCdU_B?_xvsQod~8+N7PB2Ut-tcB<w{SrXe4YKU1=?NjZA2KHZ)p`y6+4N?~(IqPCrN33-'
    '(mY9_M;kb>!H|&NTkr8qZF(<Vw9B<lOa}ZSU_$tEovV&%2rZs|!6h%cyzxOlckUu`05<H>FY<bkC+{d{X_iMSP~RRyE^3$L=(`O`3(g+On%r_DUF5|K_l-yb'
    'HhS`?j9;`}LqF`&HJ8#$a7@y{=xg>z*x*>4xsV)AtG8*{rAMHEN|AVfz^EDtBM)-'
    'BbTARqv%{z<bcO+Md*$p5i@jsOEO|`4f%Ug39YxS$8U7U2TY}wM`nQjCy8ka9h2aQp;^UZ)gPO_1%6^LZ5mh-'
    '|*vy)PgE~Z`B&zDq8l{`ZTI~M|7<X3(n_So65DlGN+j+Q7QfS9kpuls!ATzXFJUJxSsA(+qzZyuCA`@YgThohfThv8s1gsvc^86m1?UjrM8XhjwN=Hq<Wv!Q'
    'w_Sd!8#q`luh7yB-HOVyy1Snd_vdi`Ad!HES^-'
    '4O6k&E4y$&bEY+&BRlRu9d_QRhQ+nc6PueqBSKC+B<_dQ3F6@G^MsyxauS0jQsMcxC%#8Y6j^|sVuM}^=y{`9YJd$cpgPyj<=M%77pPy>>uJGwE)sI7!RjKQ'
    'BD(^w%9P8?(?kwR{PN<|~&f8AC6Z={}ZCVMo4$k!4GS<VI+HOB#S^b~aUBfEn2p{G~{Z6VCFDj)>*UmM%3)p+B_!=Xs)w_gGe_IEooJB?p`gCWf#^O+K(S=%'
    'mrk@e5i-=aqBsO=Gdi1Q)$~7MCs(noTw8gQb-'
    'o2^ZZq?YQHkIgFSkDcq{IKqN)XMO=+^6+2#G7@mdAFy3tUJ2ZULybxy0b!`#dJ@n#%GsPC93g%R{z@6_cPsjqB5TKyYJ3v?Icw1fUYKWUq+wI>ytXwc8jOEf'
    '<Jf5o7SNElNym8)v&KN`+eI1GcX2puCaU4r?rgS1y=K*o{gvlw%YGhM}x|*z#_WW$~f1TJxk*>UN7gY@l%aoi{@YyYv@7cov9TmEQ<q1>Rk6Ms)Q+xU5v8rT'
    'TiQXle)HpMc1Gb_f*rAu3dxV%+R70PU}bYZC0)ISRZO^8#Np2T3-vix9ci<Sa((EU$dUrSMQ9UO{?5T{H-'
    'kB!L3@f$$q#74t0sGvI>^eh~=}Y5$K^^$GT%cqwT#orO&*wGMs6xYH1&Pd7nmZK&?uEjoooN-Ks4+x^qVDJp%`7WEa}en0tN>S<|=dJwBOgv1yjr<1Y016n1'
    '0}_M%VJtiJZJUUt=6+X)Zs3ibx)G>Z0g2f!^-'
    '>T!cgTUFcE)S9%OHzstVcAZhiyM8Oh3aiw7c=j8$`jaYq9=kTJxxY~h?1^;gxo1Y(`=@u%UbU~Av+P1|^aP0GqspD<L{4a4y^_c9g?7|_??|@QdNh`UD!*Ry'
    '@~-dGy#J@1s42?2)ct+xMd93>Q$M3>=RG@^5rF}X*pcq6)tL0@?<$bXNA{g{_2gK+jWC-Vny*f^V?gt|s2){t){f}=tbW=~jOZ(`daqQQS6Zo_dC@36VFQh7'
    'qz?7@6sXse{$ADW9D|b_YbVX%VRWlRTRtcH`#|+1RccnB#yOqsH&uW-*|TX^dwfEzX+_(*?ZEG?(-'
    'k9u%j#{7#%5b3*)QnTj9jbCaed#{SPp?cc#rbBFb;VFifYTlHqV>J-M;;;?kv-'
    '|jHz7X5vyvI5t?V+`=;+(YW<>GHm>oq)nBGhtJ$Mh*$aH$dZlG_b(Ed6TCIA}cki*SdTvJb@9UFl{mko$YV4zCcISd;a|IH;&kAbQHKWbm#g{baovMEqOS(f'
    'dIHMNYHpuDILA7q5xtiA48sThLy+gXkZ_|NBW>i=6`o5%lJJp8*{&24GY*JlK`po|DsP1l6yHmQd3chrxa(x2ku|zsS?u@_X@d)Caf>F(ty`d$ws~SspTqAp'
    ';cIWlmqiRZnOs`{|7!~w)kFM?J67_IftEfgR)&AqDuH^On0hX%Ax>Dn}sM2b|Rjxp{I`Pi)yx&IiA{zTGu+%b*$S79DwVqf|I}+G<yQ-rbZ*dqzFv}Zn-'
    ')T#~w{Ygz_PW(?9?dd+_Zw^c!l(BY_*qUPa*5Sud%P4MbwTBL@6M@}Wg4?S&f7YblK_pZQ5~;p^O$Ddczz?E`W5TB5?{m~WxYn}N~>={Z8bvus=valhii7JR'
    '*gwY<@bO;uj>8@-CL`1s#M<(bl)qhp@-APGuy3JOk=Mfs6X~;-_`nE&DVumYfHONC6sA?Y_V-<e4<!3dHwFaEe0N0sgZuyeN8H%S-'
    'pyb8v4A5sIO1FG52cIx<0RA%@}{1)+n#C>!$QRJZeqc>m6ED$;NMNGv8?)g~25@HHt?6k2RZ)gIuZ)#-OfMo_*ss{qu-SvCG){>rnfLc`LH|eP1>BZJpNm&Z'
    'xh)dhUe?MZMaYQQPeWzko_F;nlsVtX=(8t*3@m&#1<-9aKKc?qUo6Qr9N2bz`bAtv<!T)?PG!OT=BW8lP%Czopi;>hD>N)uqa<!s4;-'
    'pI6(C^gXO6@3jXsvku=h--l}5u3Fm<KK`z<jA>S?4LPmLSB-nCdhi1N(4$d&(Z2(&x(=|%EU}`r#%)x0PT>1*=(87H-'
    '_RW8K#=>u_`9_VefG?&E$te~Q~l;Qcu3EUs=g?v!ko%_<RoigbPpIC`(fLhBesNddTIfb(<tIPc<`m3Gd}AmfRWF*dbO(65M>0;HP4P|mEft_N7z(dU24OKu'
    'Gyo!Q7c>2^KOmJEqI<srClXls6Dn6Y=uNMW}V<RVU6ttF@`$TIjuG|s_Y7_y%}9KR$a|I)~S*wH404{*JgHTBTwE_c68Ss7H)}Z9>6D<;T(KpcBWO<A*b1nz'
    'N+<fyY6-'
    'LrbB(Y*I#d%|5x_3NwsK%$k#pA+>X}RzG^GM*BsGqk_2ZdrH9j6hl_f0n+WDJmR3?DI<8qTR$<R~my_IHRl7##MXR@jmA<U;YuEjLdq&khdyOv|uMF$RyI2{'
    'hu6NL9wX2pD*TOs92DaU(kr>13+Q$E%(3?81nyxek8I6|l)gJY<Tw^uIeq>AAKKdoA$dROHJ(I=Hnqs9H`Oo4rCA2omRaRj(M>QHdDyvc@#Idj2)grH@3q~`'
    '8znoS-'
    'a;mvXEzGHoI+Ze|>l2!XQnkQyZFJnI=n}~7k>=OY*t|+z17+C3c3A{v@L6h)^o+G*%=d)1Xa;nwhIm1<O59ib2USxA7HAV^<)K=XQE!h`o>Bj>eqO5e+nl1_'
    'Q4SdO95|h=+EF5#g>%d&>9ES0;$(|pTdsrK7<*0-_x4Fr#Tx2UeQWyL?d#Q>)T{p5k11j8463xW=JHi{$5>SnaDYKwJpe(p_wI<CJ-'
    'T;|M3Qxt=3Ly=^+%%Hm7t<E8ozhV>lh<%8@W>Dws1BstF?{bUiYA`r5cMcc=i&gtv#RwF&-nGWh%pwj3JeKttWgYUFefmtqmhU_FFD=-'
    'RDG!TK~Xq*`fYD;venfQADu#@>ut0yp#7J#XY)Or;)eKY0t)3)|&e78G6){o^RV>DUcE)Xoa3eI3>C`xsO$fk*^Bfn*uG{)Cf)Di#WnN!A^Upeywu~dVKRme'
    '`^c0BBn8D(38(ZF1)|k!ZNznPqf2(`h&)6pNMH2CwfAkI#O<{;gL6PM87Yq9ey`!h_K#)qxBI<X=j%(ZWUK4jzslo4Aa<2DSdvYC)+e~SrAvRjAzw3t9@bvJ'
    'jwoa6<i8iO|~lRH`~H|W>>clajKF!b*({lENO;k*%3BW)}+S2M7_$Y=OeoBj`6!@W}o%FT-Vl!%jQ+nq<UgIAx`{wO+On{>Kb_J9f;qV#%C1&uu-'
    'i`u**AccdWa1v}VUu%7n^xE#>-`RJo3zJgFCs6W+0>9Wvv2&CZ0@Tcti-'
    '!osnCQl|UYH3EGq{Z!++qxDuv6s1S~UQ@{nT5*Y@M`<MIfs=X#gnbeVa1e{<omf*Bqgx5Tqg`d4>%Jbm;d3yBCVg*IyQabJ9FelGWTbabW9n%BK8RMQMj@vf'
    '8&zgn<L-EC1<1s@R=uO5m-<@M)dwO}m&A47iMVuXTvjy?-'
    'f<4pl1(Biy^KOuJ#JEcw*QjYsbg3oZreD_gkiO7jX1(FsKdC*SSKF8OXPJyGwJus(e^vgE|2#WILL-R^Vwv-'
    'u?e<KH8|g*?zkjEyP)=NQ<B%TnFnoP1S=}#mi5%H5skt*7*^kJ_2oJ)G^R=QX;3X6(04O9PBi09Fe7Xeq(NOBW8c*MO~ecx`LRW}rq9;do31oNw&ZU$HU&;g'
    'N>{vx_!NqRkQhyH6d(=eW$sIr{yB0|#vXLc+Fei^Js$n4wVMc6wVs#+ONyy}BM}3{4vj6D%j4*)EtLfJ--AZ4hw*sU82NlEQypHjmHIrP(Wzv|U)Gftc9VHn'
    'D2poFaiuqvzKdT}sXiL7FDH_k1PdI}U5+O>GPp@>#F6K2-ePk+E|{e);&#UU2Q^wl>VfUBW{q=-h*m#)MwIzz2c<Yxsr~wUj#yCxtIUzBdHm%E-'
    'dD$y)>Nmx;9-'
    '^gqLEu<^bTOLIMULn)^B0E)`5d%bcgpz`#E9N(Z)IHlhM1{ipJ8r{y0`u8?|?Wem;OwG^xBNcFhCECJV=@R^>e6|IMhjJC!oSPTP#NYo@}DS~|+RyR9`;s?p'
    't7pN_S<yo2w9JC+h_@cT8%d%mRcG?%Li%)${&pD$^(@>z4TpzCG>*>3mSd!p-'
    '`thp&pk~7`?LewQj<jp&`XUnJ2sQP21*l+5aW_6qBKuk~Gz^Zz{zi7~RV^z;^K_1u@moz)O#3ygG(vCn@SMc}l_17a&)2Lc|qp`KM{7S6MKKwpWpmoi_iRQ0'
    'gb$QLUGCS+Ulh=v%+cq#l9fr%%OthvCtY(?$N`*?<!m^F1>_@C%+gg6dUa-'
    'B}Ibole2}f`%b!}4Px}cWKfa~63FH8`zHu7?fg<lJAZ$xv{jwj(a(PyE#HB<U@1b_QTB{iyidwbKc7hcqxA<gnKe!1h0bKtG*8i_${ig}_!@0x=s{cpwtv>&'
    'v97u%w78&mIZRL&7tgLmU*cpMG*7<J73D~S1F(PFUg6<5m}HK(td(`GO_TO3(kDI><7g|X16x}M;@)vDD8Dy2`?d-RD}JvXYSMfExsxvst>HB+ZpbH-'
    'B~LE2WECRN%BT!nJjjt8*2PVpVi^^CEijGpw!o)FVCL*hpLbc>Dg4YH4v5j%bc*BI56Tf7iQ{ElE&IkItt9a^t89I5;+^`?;>)LzB}`(cO1GY>v~q9>zZ5JP'
    'Zze4bY_1DBK+)93cNHdSv4YucQ+UahfPm5|ZB1G>-'
    'gD#w$K;9S<hN6W!)sNy_tQGaUj4$~TeB;3bqJuwVQHmf=&_45T>$G+b#czGv`8T*j_m(^drjNu#HpL%SJN0@z`U@JzqHenXcXq?`OvEJaD_pnZDG?Jt2{*_>'
    '~d17!E;K1+TUq;RL^_*h~cbsglXF=_2SC71tPU}s5RbN`*Rpn`yt%GHD>IVJH!5$ig&2om_(~W1c$(**raO#Eqlvc?RkkJ7wn|}5wa|i5)Uuv|B+J$w!inUp'
    '(_IQ;VWk?WHU(vIU!tLqj2=C4aJdkRQjCl)_`ssLYL|6CJ{~CQi)Ri~9q*L{`QRUshz{p@#4e6gf$qPN<H`V7rj?r{1&u8#645LB3@<B>}S1rZ|N7R;6&^P-'
    'VCn}?j`Kuy=SEYF#BqlaYpS)`gtB$H-'
    'T*|zpa$R2_?s}pt<_A3M^G4X82k;68^pyfHxzSoYhUMC?F?@!{bj(}%sP!<<sW*fFe?h!rjni@f#)}c%6lnASyHp0QkWY$4k*Sl>Z2HYSBJyTzEDJ7^04FvV'
    'CQErn33`cDI9hH_*)&XuILrdau99j;Gi>7nwYo!B9ixe<6yxTnM1vOf^f=h#h3;%(zc|(ycCdR7!rHXmvZN>Lbk{4fL!-'
    '+hYJWMa)$xioJv9ezV&r@RuW?N+bS!xr`(XpzWf1H*M=Zx2ggTXBp2m$@astA($BsA!zR;+(TLOZ!qn0_Y>L_*w4{=yERj34GrCGJ%9bYl2r&rV_qv{tb`+?'
    '}ovF_YqotD7#c_sQ<rFHDJzXk_$UcWhJn&7<ZQB5TppB~jzuHRz17Qw>WQ9J51@8#?{bG%nB{ft<f=c*5n-4@Xkv7u{F%LyWuZ5pE%(8+qZyESl4hBQCs-'
    'X*XO4)pyHw9@0XkGDOf)(>DGn?-'
    ')7=jv5=k47^G=jc`?nDJremKo4K369ji2WI;i)WU3@1gvf&lMh&!r?7LgFs(1(VoYnFOu?NiXC$*4Q~Lpz#9Dh)s`o~hK7=pYpq@9ei>?thEm3PGSP|wzo`I'
    '1zsMRNU_%XGyUL$f17t#E(G?-'
    '2kJkwiP1!ghX2bxen%tCB~LuqWm+~t@`9pPQxhaF$7Ye{Of)$P;fhPT0ZYg$kBfWU>>B|Hb68r?e1lwIywP_Js(W9(O$LmyV1VbG_nN?ZV`7|<O@>~Bj%Rc?'
    'zB%vB<3j;Ws$wL4|!UdDq<V%wR+WfovqEqhiO!&nTCcsssrp5_cZ$#d4Gc^YPkW#Nqs!q~8VlVf*Zz-'
    'uyUZN&NtzGo->bIdml60(4uunt$S2Roo&Po|07t`QMC)z_5fbBz71mdH(9t#9C5wjb}vkiCc@_=z<diw#{lQRzp-RO4!M;iQ_z)-'
    ';pLIGXo%bD9>2EY+&kRdDH2U5^m4HeQtGO?w1m>r-'
    '3G)azl^=MMK|uu8Ud&kR=e3mDRhKDpF7x+HF#)EEqDMb^S#t5e%joQ@B$uiw@G6JkD3Fe7XA*BtRw^POYrO%Jw^IVVk;g+q<pjP5dXuL1nuESXg}gE3hB9dM'
    '6rr+8GJb+3`K61Z9B5+p%x&BW?f?-'
    'pT{M`307oHvWjdv1x=qiy$jwfbIvn<M|m?s%(NDA(%iVddrF7Mx%a$Fw)yf+alZlLOt=4a;(oOo1Bwn1t%xQLlYI&*&~A)V5_J8bvdW9TiT3lIL|#C-'
    'JmC{@K%B1oIu&Om1MwI!oe+c+EImKQl#+VVsX}h99V1dm6_j)!u;T(hOR-'
    '3%1sV7c#3cIlv}Z)~9o13AEAoCE}6Y@RcWFkF>%SyjBa1>pa6BZBtDiv1zah&r1@XM>UxVwloTixCmC+JXYH!Z}utNw^z>c8!|xpIGxv3+pKyQ)~t9&G9Y=5'
    '*Bp}p^8l~79E+@4eVGTLHVSgVj_>}T=xS8IM~E65$FwJyR{Od%PH((HIj!&%zMPrAXT){Gx-'
    '+IbFV*@oA{Vxo?CT}SC3?bht<fDz#Hj6;@9K#)jnoJ+pec~cULtRKEX98KH>2bOczlLn91g1dT6lE_MEa8Y#OU6;K0Rg+DucgZ&-'
    'Pj6&4H7cpApjxn6KqMVv-XpM~ulF?Rh)_=Qy<DZJIY!#R!!!4`#X9!f=dzR8PF9H}iPBn;=NeN3sQGFJe=ljqxtLkj0V4MtWCmBgCK`A9IFIhIn%l4$Uz$nI'
    'a?0nLVrQs2O;U2iP9dpkVhf`4f7^k+XFomd-V)(f4WnZgj_NwkdGhO)?PfVS^jHyHTmL>`;!cy{Xg-'
    'cA0u`@G6j}O0DL4^>GLbdPQS*jqlP5#=ESZ<UkO8Mm-Z_FYG2=s?qPz9XwTI1ILWeIlIat+_o-'
    'G%VXF+SLEl6sWqLt_Xz&Gal~u5Ik!X#ma(ZF?|dVJ!w6{$Jey5oF}ASA)i-BtRI8`XW}4J}ZR*b*{5M;C4P+i&st#i}W^1M4;2AY+V)rQUI-'
    'L)4$@*!=)*gio>9^pO)|8VE_f9rTlvy9rh^Fa93^vdK^S;Np+FR~s_qri=^J}D+^)Es^auA>60hD8!9Ed}(<K1E&&I(K%V;beK$13%-'
    'v+|~3aToUGc9;}ZaLJ7$52?Middgnr6gev9SBzpgtb@iGN$=5)_O5GYmpPZt(Mz-1pRl%_J@5iY#<P&r&rXf^iAwkS--'
    'nI4s*+07B1fJbrFhrpCFHW1XKSBhQ}?dGq}o!8I@N}RYThQ}X^A+rBPLsL+|0K#yI=vV>jtaQwo)@$1Qj4`&P_YhiaP|kj!{RwTJ%QGN?_ldk$+WS-I_l=US'
    'um6$7q4S`KGZN#o}Ecx|f5!v`}O)ufexyhJTmEA2cuTMm=&&W`zj4eO>2DX4Ee8nKD`>m7uU0cKkdRX<ol&bZrW+C5pAYOZz8OVp?mYStD?dA6%iDkH~h&fy'
    '~>MX$LVKgKss0H@{CLx(3T_0{kJWzO`r-'
    'b;J3!Z+L=D<_MwRt!A>emem(qu(lkUIBNz}L%*Jy(Df7D*Gf4BJ3gjsE8q&fy894(HUh_BTh|WM(wIgw4z@J}1~vk#A_ijA1sAuP%%p3g#O9s8=-'
    'yJ*l2;qYHLkPZ<gY}Ioe!8)TiSKkBtD}Nglpa+Tjw?GO9yzQj&^&ee22jq*SyVYCG^6ant)ZXKrg!0%3k~#Gl07^0!u_e(^@6(%u$*4g?s&OG$sMswWud<h^'
    'x-%FK5<8v1FW;X#3e&5HUQFdv=jU*!6Fk*D_YDV}EI)7YF1Yjj4|rID=#CUVT^?wtHjz<_tw6?$dDn@|-8-'
    'aJ}*{g%(+RDe`lgh%=w#e>pb40<Qa@e^2b(MusPdts4RJ`r0NF#d|}$${hzYc!gts4ok{dpmDc;-'
    's3@h&MPc9GoAOytm%UL*9kjslv5!Gf20K;GDoDKUS%08-'
    'h`!a0EWD+I{c|4faO{}Nn&JiIJaijJ!4~DYqpwH!<JgVr1EUdw5mr{#MK7D{*TnEakzIKoSKtl?mlaDO2B*GG}4`_tCtMLO{|JAF}oL-'
    'Y0eKlq_l3%6vw){vA1I6JU4><WXP0mP)n+{GS;;svW#K9N{s1yJ9v`0P#OJYyD^IoY3`2qx=C_j-'
    '(b?LfSo5{G7Z5ua?a8!ywxU7@j<dhhIRFd*m)Bem)Q?{nv*^J+Ae+BB61us%lT!N-'
    '2*vPwk%f3HuZaMtLRyGzp0;9S|8)AB4_)?uy?)McJT3bIWHPCR<_~mRK7Ev`aoP-VeObt?YF~ks#nz_r&J}e=yvv+RnEF&qF=Sdp<+Zo((FtzqM)sC5{--'
    'xl8=)mcgX)Li5VWl95FMwz=d=S)V@j`?a6RXO%daHg`0W7`{I6Oif_p*zDg@>60f;t_*(C*)^=9tm})KMY&9P!tKOX82`;LRE|{D3c;~g)pT@-'
    '2HIl~5YQYrG;h!v%onyXMg<7(&nj9r>BPN}~Kkn1t$7;(KoLi%_Wm*m9nMANQAN6TBe!bt2IOwjCt#deN%lf-'
    'U&#kf(?64~ro}6Z{vF*QuM{x|ET!Ggz40gJ%(HVrHbqxE_oc9h!>XDofm+4)nznF28*Z9tp-'
    'y2s+MzKoxbcGXQSS3{x=N}+$XW!}(er~y1@(SCtjukZl55pX7$CRev=Dq7(X@z$<SY*tatuf9B`24#k`qd3X#<Q`)J>_amn^tcac7KJ+I|KV#ffwNTN;hk9l'
    '3Fgvy1M6_u|?EJ1oDit(%GM_Fc8nwwixlcN37gYSnWnroULRuYE+|eL`L-s7~TkZUiD;XY=RA12f@3ccg#bwiNYGaOR(!>-'
    '5VjZqK<q4bC4ET|ITS!CK|N|OKOu*ENA{FiNV%kPml3NI?s25-E4;lP6o!Lee4F_=uQ|>bNDp<SR=jURu$qV7sT9*er}SLFitdl0#-'
    '>U9&|tG*C6Lygq)xjEJ)*{*X-'
    '41xtblYrx_oILF{#D?sqQ?+8fo;p%xqyU4JQNq~vtZ4d+lf(G>f1b^7VNC}Yt5c&ahRs}&T%IdKDc$;RJawaTjCEVY2)4e`FO63HAU<7A#p{C)DFQ{)A&6ZO'
    '3&Yb;C7Mz317rgd`08<b=PdZieZtOqBwx86o>ODhre5j<FDyzb$B7{l|)e@b-yRlPfgZ*Fdj)i6$qIVwi-'
    'Z6h_32|A8fJjKpoFTp!g7qx7X#Z*c}a~}Mt2mVtQZ_jU*&+=0&57&AQhOh&VagI~)NGs<8-'
    '!c!PG^%HvX>&^6b*b*}0N1Q#pMT=rIt1H#h3A|jno*)w7E~;3#l$uGZy=P@<nFz)b2<a(8P-IZ`W6NMHrMC|)U}m;-'
    '<+T=u%9w8geJ{M2idaI>Z_5pDz)}Ncl43#=hZR98E~YQ*w>lVyc}p0S~zdUQ4qN$i~E*1VH8i+v9S@N0cM+(!vl?BQ>=i*G?2mGO}4{1V>zr|52Ip{$4)y`?'
    'I+|a$H6+h)4u7BxJJz^fhCx9*Yq*1J7d_GG4=nPQ^6?JhT63NI`hO1V~gTJpF4IvME+C~%gss-'
    'I~v~+cz})$KeEPsBBqI!Oz>V9<tycM?8Cp>Q7iXA?;gn0tkw+NVf{zpc6X?M&f+%?aE;B^rnWc3`ZoK05({V^u9u@Y&oI#@$kl&`H+!om*Wj%lqgG(dA_8V<'
    'Ua8Ul8a#w;(Bm_(h%x+_SGB93Iqzo2?&nO*>FNgM)x!K9BSODHuHZJ=6k%q{d8RG2VgP&SotR-'
    '5M8Z~P1*fa|Ez9f<<K$_*^S)KWCppzBIVJvb1*TTbS{WcqvW@7?4!&N7xk<usoWRZ-#UhJfH64T1%}{Fv(WX7PbLLs)G&U`&f15MS`7FIeT84S&SLum8C-'
    '3BYYMpUydqca}D_OKDo;f{cum>iwP3qyNbu%_wx_61}a<f!y0}hji(?Op8G4a|MW053Vwg)y^n8^83G4tIBm=W<?{BP&Ju9EFFj#rydFYE_(s@_$NQ;MIq^i'
    'x`=DZTZDd>1RZyun2&>~+IfF0C3DM;u;2pC8~fH|p~z5S<D#08YurI@1neEU_D99p_RQH+_MXvBTNAM`mX$_~sKk<`78hpn6)Pt2vGP74Mm`|0NiMDLqw*{b'
    'uciYB*KfL=WH5VTpwT{h~m>D9|qo^os)hqCmeW&@T$~ivs<kK))!^FADUF0{x;uzbMcz3iOKt{h~m>D9|qo^os)hqCmeW&@T$~ivs<kK))!^FADUF0{x;uzb'
    'Mcz3iOKt{h~m>D9|qo^os)hqCmeW&@T$~ivs<kK))!^FADUF0{x;uzbMcz3iOKt{h~m>=zm4ODE>{qsQN>{DE6UWG>h$;Cq5o6>KE<&s9&`4Nx!H<Bm1A|7f'
    'pa9c74+?io;wg=odwc`bCLP`bD`v^os`3FG_ys7v=Q(;}`v+)*tkXj^R+;5yAMZUo=(JFUsLNTED3JlYY?ySknWT<)8XRW~_L}@BC4}XaohCf_~A$U-'
    'XOO=oel7MZakBi+<4_Y^;KQQTt!?i`w-'
    '46x*zzU$ltrX8od4@;>|)^b<j8)RQf$ub^L4!ydE^f8q6ue$f<cxPQ_wawg(u{i0R!X`kT{{njtaedrghf9MzG|Ija5W$!R+s2YWVU;0I1kc`jzMeWRrIW#x'
    'LoZ2Nky5IUmQ|K4PHE!$0tk2K-MNJ?2MNyUS9LXyf51;jmdWrT8=#Jm|MT6)UnfIFoO+7%5<+py3^H>(h6aJ-'
    'N^oWMRZ~dauAM}g%KlF>@SbRV07Y*V^U6G|a1|nGn2d$UcKmE`z8u+4L)Kt_j$`$pC=05a`#yQ8o>K7RqSS_-'
    '1KI<2~{nz?M(LeNy7D0dZzvveoe$p=*3>EZ?X8$AoqI%H4f7dT+E$SELKJ<%rKlF=?0ytCrp1sZbMJe=)20!$RM)aKf?S12yeo;AG+OPUW@jvv72EXVRRs5h'
    '|^hOSA9Axsde$fj0Gu40S7sWs67dZ?0XZ@l*^oyFm=odLB!wign)-Sr|={s_4Tkx^I>lc}i={;<Vwft}TMNw2&3i?H*Mg5`+wdeuf!)N`Xa8bX=Su-od5q{D'
    'y%Bdv<{i0bA<O=vj&)Axq-'
    '}H;d)fVd)CD{dc$bK|7@?F2^g+6}QFS`AtUlje&FY+GtOTTCpEs#?Fs)c9tr+!iO7yY6>V$cQsqPWT|=oi((*C^;0xtHm0`bGOi{h}@6B472320!$R&h&Rdz'
    'o`93{UTe+r62l5Wnc7*_P^;Dt$ouk$`<vDDqx;`*DtF5re8FyUhWa|9>ALTrC+p5l-AL}-'
    '7oq@QS^)AMg5}jZ~8^m@H!W;dxtnX>vWgB$j|ykxi9)f(V~7)DOmSs{i5iHev$JB&OnCmi5nVWT_?l7pkEaELBGgo|1bR_zsF5~)h|lJwEV1JWOO<P=lz#{Q'
    '51EZ-}*(lKlF?8-}H-iKlF<_$U-'
    'XU7uCa}{;XdV{?IRqf^ir0ixz*>FUs@g7?)2J^^11D=oh7m`bD>>IXL$HRln%si+)iIy!W?$QSL*(=$uwL6K)zc!22Kdi=3^S!(#lcUlawOpTcV0AV1Pk*Pr'
    '!^nm_3m-Ee-HWw%8RRpvv#==v}EMGIv6c3|&%m-(e%wD5<1(di%hMfOq)`bDki7nxnv#wplS%$1CN(Jwmuq+itjNx!H`H5T-'
    'X;_U3d^^3YyYWzdLXyy<7qOvdgMb06uBfET(e9qtcMdpS6(l3gkijes)^^4*#fD8IX7e)P|0$aj<?%(u_az*{3i*Ncx<3umYh^uVC@?3(?@;CjW@-'
    'O;D&*TL>u#Z=N=od8<^@~bB^o!DJ<@Oi-qIzPn1^uFjPx?jX{oeeGeo+|xqK2Y=QT0FQ7sb#_$)R7g@<qSs9{r+6GCMx&7d5~R{i<J7TGTJP`lerWq4Ajcre'
    '9>==K_BFXZ<3t<85NX_wWFJ)-'
    'TF_(l5%<?t*^N>yP?HxexuK4jAON;C|LGx&!Sl=ojU_=of8%(JwM1DMPOwU@LsqFIpyt<@!Uv=pOt3mwr+DKhQ6_!S4R8U)20Xzvu=1BCq+q5B;J%r^gM7M1'
    '{S&3ES}j{h|w^V4wAihE;w+zo`62{i5}6`bFt4`bEx4SS+Flf9e<I;4G!U@Zvw}7e&x7vVHJNzi1t7fAAmli>BC%CO`Cx;zj+Ucu~J-'
    '??b<6Pf!2WFFO9vFPi;Bzv%S`{i1608O&6)e$g&<IEt8q!S_qQXaIj`rl?;OC6l#<jNLr!z&-Sf^6(f_=q(??+j=YN7v<0|S}E!m-H-|LOTVb0s9%&;-'
    'GAyA9sCFSMXhLFWQ+PmWkvm>XY@l(K!ME(*<xk?(l6RZzbJ=((d2)vUo`xqe$j1Fzi9W9eo+q`v|su~?_{vg74?f=*rSJ3W<kGb_ecGr(xQG*?u&j=PG3Vs{'
    'h|xi@>RcR3?Jv3d};GA3i?H(#C8k%MeEqu3DgDb)uxKk;eviq3JdtNeo?2|m4us=;sn_Pos0ifzv!i?Uz9{k#rj1#u)bOHPJhxbio*(jCl~Cue$mJ${h~+qp'
    'WpgL5A2-jAM}g5i~2?DAhDO^yKfMo{!_mw{-IywtH1P%!e8`@=6=*K>Q--l=@;#P=ohWRaQUrYwDN;~(J_eYcm1OB5B;K*5B;JO;w8WIi>ANn7u|9$^pa<s('
    '+Vo+7u^$88~8)N$c&VNevvW5@A^f9pY)4*Q4V?jQNL(~c<m!Oq`&oxHa_%=lI)d6z|!a!jiX=GfPPVRQNPI8cn!Y!SN)=gqJB~IL%-;lyr<-ceo-em2({poz'
    'x9jGVI_Kx>(dN0!vz0RzbKD>Q8Rhm1^uEIPSG5g!*~6n@=y9jM$jA!zd@y~pkLIEKk`|>Xbl8n2(D3}UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%p'
    'kEZ|7X|u7fqqe-UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7X|u7fqqe-'
    'UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7X|u7fqqe-'
    'UliyU{jcd4ZDGqqKJ<&yYUeNgqE6yzeME|W>la1;&@VdCGh0M>nlwAZ#6uJRpkH+Vp<lFweo=({JL?zCebO&l`lMfU0DEK$uHZE*zMx-Js?Waa7agF6@UwnV'
    'g5BkB`bF_i`bD$q-B0>OVKBI_`bBL;{h}+~-3hC;=`Z?4TR-R*)fe@PtY~wKeo+klA~Tr&O}{AmSN)>LqJGhSQNL)hs9%%^9W&!_P^;pXevu=6$-'
    'n3qJ^i6yRE>U7_z(S}di0A9f7CCU_@-Yp1wQgszi6<iU-XK8(fL2<7p-uve%3FFgAe@DFG_vr7rhkqi<ZHp*T}Bx)I8k7<o~T-'
    'wDX}~R11r#8&#CYKlF>DprgO^i%LKAiyn)rItR@5*eCs>O|qX?bXT;fU*w$3KlO{=e$+2o|3km%`9r^GAGXgGnC>tAqKgmxqU#U+qPHLQi&~2MMX3+{qUb;B'
    '7d64>`=wu$U_bt6{i6AQ(J$Hq{TckLe$if0zi9lw&@Za^NByE<*kc9#qB#0RN7V64zvu}rx%G?8<g<R!9QsA`U-'
    'XNXLF0bu7g?#|PyM3UANobfzvvg$eCQV?$fP|Xy7FDWsObm&qQ;_rkul~iIBvi7i>AoNa7K`MoXunreAO>H{8#;=0_)cN>G2|4;pm%wk+pHEP}9l5UixSKq8'
    'rxeZ~dY-_^ugP%vC??7bVay%KSyYXylWA(Hf`D9Qs9X@N&ND7iEh2MP)>;8jJcxyI=H+-Za-9t3UONtgP`%zo-_?lumMDtzVQUH{ZE;t=OXlJ3b1-'
    '?W=xK<ePrc1yPDua>yQF5Eb-`7K-{so9fH@Kj;^k!94Rxzv%Qszi6|lUljch^ow@U-6-'
    'f6z5Pe}MO#JvBJ2C?qP!9P&@Wp2reC!Gp<mR2{any5s^_^2qJcvn`bB1dU!Y&q`ANTM^OJtj+YkChyCCkL^^3}hN&nI>dj6zebpJ)aXyrq{=;@1oQ76pQ-'
    '}*(i0e<Tjy%hC}TEFQR4SdrtI{2htWN$z5O~0rGAM&$)(F>XwSLC&9f6^}+N53cu-uPR;==npxs0qyKw|-'
    'G6t0{#N;Aj0J$2gyetT}V{vwl$*`bAr0DE^av(MwUk=tB22{h(je`bEDe`%S-S>M#05TYu3nN`fxi4&VCFFS;O$;J1E}*-'
    'l$U{i4l()Gul(>K7ds^@}ReFN*S>|JE;R{Ln8d*B#?T8~$CtXycoH(fc?3q7?c?*XS38@dU4n`bBHRp`w4)FDfnS7iB;6i<*cTF2fut=of90XZcmXX!x)CMV'
    'aDT(puCndV^zR{i2>9^ouUQAD;d}zbO5oU*tSDYhyIRmT>M_;t&0zxuSki8~8#B9f!~QMH$f9GBO4V`bC!?`bEPZ`bDoF`b7(W=obyZn2mhWFX}|U=<$<&Q3'
    'dhFU;0I6o;v!I`9r^`0^aZ2kNQQ|f6*_B{DXc`BbLG9Kj;_j|3$y(fmZ(3FKYckzi6E-=*>U$i%!5TKkFA=!9m?6Kk>Y%^;gg@n);+)l=`G!bPt-'
    '+g_gi8GxS}*Xy`+~X!$?TFNzfPi&91XqHs~asPvP5(V+e|UT;sUTfI7i3-?{W=-'
    '>zaqJ|IsqUc}ri!SufF}Fw9(qHw9cFD}!0_FLvUv&DRU$jI{&Y${4Pv7*5Zi@Ov^Z$W<QR~0x7rp(UU(}?#s<2gm=@-'
    'p^=oeKN^@~!U^oy>^8v9efX!k?EsI;hGR8E|&`I~;xIbKxylYUVj{O#eQeo@sA`b8sO^ousoFN%E8FUpXM;wYj|vVwk5{U7>8ezSh-'
    '7tQIubJbwa+&soPGOmv_7r*t3R`uNO7yY8lhknuF7yY84AN7m2uwo|AFKU9(9RH+WbcyE0W>LRru&7^jL$28gZ+T2l{Hb4bq;@z{VUpGSpXe7gqF>bX7yTmZ'
    'Bv{QPgO*G|zi9cBe$hD=<!}9>C0Koj|Egb<K))!9_VF+MqLv@@iyGlh6!eQ0KxdxVVLFTYMUGuBedrfmf6^~1;|v{8ZwvZGb3f`A4Z=;D(=!GAq7<w5yMEC%'
    '`bAGg{UW!t3H_pGSl`~4|I{zC(!&I~`v0U~WXv-9Nx$d`yKLw`)-Q5-f9e-'
    ')f6^~{`ip*1{g3)Z{h#!Uj$tPKH~K|Sf9Mwt|3$xO?~8s>(<l9+N45Qzevv(c&-'
    'z7s|EOQokL7QB;G(EswEs!JXcy&>qaXB(Mo?dAg;(k*+%Ns2)(`!n!4LhS<cEIIrb<~Q+v}@-QRf%^qGqCX)-PJ;blpO~DDh3dXricJ)cH-'
    'nXzPo9Q3XEbzv~z6{Z+rH>XUv^frGK2-u>1u>ia{#=mySvpkEZ|7X|u7fqqe-UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7X|u7fqqe-'
    'UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7X|u7fqqe-UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7yU2k7n!x{406XrE8(K#'
    '$?}Tn-z?Gfdg8OYL~u@Eo;2(EEMJwXEvr^v>B$Y=R<El=wKUJVE=1KbSo`mq-'
    '<(Qo1Uu<sH)+=lnjJc>H~NLx@jNx0sk~V*%4@hL7wp7uaML3&QChJq&HW!DLUTj>X8{g-5`O44e0aZ6JMhxYUu=PU-'
    '9=Wub4nM9#oF(ArS?udmpE&>m%OD`)Mg&RRd)1T3P#TZ9MpZ3R+1=Dn1zrauiHI;MqR=7t*vu=?<+*k_gGaAWCpCEFpyB|Z^6q?$#dPKFU}W!&~ITF=kxHaE'
    '@A&V3v?ZB*#Mciv&2cJ@jXiMVQMr3X74%&V~kv#Fz3NNIt#T#5sfu>gM>GraZ;*QKEi6+1(RroLv2ltS9bV*tg)=_IV8VePpzLNE>+0cDD2%8WKZ01&Y4GS)'
    'u#<G$Op}IGkFA?L<ekz=E0b}Q}4mOYy&N+g6(C^Cv!vY;cHali8$*bs~S%A-'
    'j(Yq^Ih}gj%=v6&a_!&Jj~}m2dgpGc?eQ=f>y^6nla~^&vnj|S=2Y2JzWXTUq=k@3P$A%x-'
    'Q%7tmeTd8R0Xc+4roHYP_vMc#u)G`BZZ`t<UX2IZHANLvywmn@r+C&uQ!(V_qdAB@Dvx2m_@ZzuY}}Mf>9j{N0gtw;_k-'
    'f@A%+uq+*~ct*>{wwtZ0Wn$G<9DC7Bc=nAd-ILcJhILX+JZBD`-'
    'W(%t{$)QIHh09w=Gdzn$=ESsR|m4a%6M*o3)_yX;*M%?;erj?#>r=fpz}GBWCiEYq?$+P$T6EZnPm~!3)VNPz{}cJTi)S3B*58>^bC=^TB~}tK`48PjM_Fj='
    '8PQBlXKwbK6}={_WQsEjklN?XwGLEB>z&QP(sZ3nVj_u?23MP1z|WeB_M%AV3%fJT#~bITc-'
    '@HCJKk^1;+R)N*c2$Cz+G8$vJ$9TE`KHrg@&V;AdsJcE!%?ERPu?LzQ}>1}y^fO^h4A!2Zc%r`D4pI00K}j#({Xu4dKaI&#Dg$?7tu>m5I)`@CnEFPI^YP^l'
    'hH5>;9zv-'
    '=Q!*f6;N8mygFn2#^y;|`;>WrV%1n3HpghT0&kJ|kn7WRka&6Jkqz3U#OzP`4}ftQ!`UaTQ0NFX00?qP>m&*9o3;Bl=RdpDN%rrpdj3Cs!}bK4%7%Jr{FEjk'
    '-H`$M%^s^ZUu_D#h*#g9)2=mSZ-'
    'T$RXI!?1p*Q$~iln3wuiT?*^!)*LQ?$i8_=8Qm9yL!{2ireIAQ8L8jzAxQTtKXL9+6(E)Na_m*9EQcpNTv4D@&k^yzaE^M~(4qOqhk$(8LCmR1OdJ>&#ZAzs'
    '_h>16%sL@Y0<fJ|^4`PrAruW$%@~L*ftvW#Pw#clQCO0g`TCg3~h40*e71c_X>$}E$nYV9}Otg8<YiF*vvy)xmMc%=@nM6%&3!R&NvigTn9B}mT0FK=$KEWl'
    'u$dem(h@W1sHq~KGu9D%HChocorqoaT`iRmy!L-'
    'eqK0=A7lN_!}d;#OHWAO3)e)bZXsmB^MH`STtdwOyZjP{a}oTa#lm0!?_%c}MrvIMTk4%lRuv#;F9T#lhbVYc%sS+7k*qZUCud)0~uVuBsy?(LG%k=AD;cx>'
    'b3{gk7(H3T=&9?m>`$O&>BI@!VZ(Q+ApC3T7=Ur8>+EBa@%oUy0mBhI1fWN*@3HG6_XoSN5UO*iVfGT6Gd9jw*&PJV{ZkarlbZQx`X;wZ=1_)~cPx3F_!tU6'
    ';_>sYOh50;^A<$R!i?35eMQ|BVqk~?_`-nYii6bBthz);BQ-XS8TE%1|X!RgmoJ!kOETv{EmjYH1)Mt1LeVr30vR!`#bHR7#Z!f_rX)6TjRqpGbI|IGG}`4g'
    '+4<M-$^O!N1FzVD%QumPU83s%y^ZgRl+X|0<E&U@!<J@c;koSq>L?Yys9bRJI`+f5kLqeO-'
    '}Ipqgod%wcqavp3StLYW%V;fB?^D28_cs%04Ilsj5uvan+ZeW{lgS+3ewwCeAs)#2x5>vW|!CC>2aTc4}9?AiJ%?x&s6&@Eg=T~5JhvaxXG8gsmW#j6-'
    '+0v8vsrCj2b>&9AY9$A64sQJ_5&JIe5$Bn7py{^6p5IAqzX~p(t%ptW1D!MCOu8M_bB=ZYNZ)2r#+-'
    'y1S<k6lTGZI6BtPq!Qz8cDa7KPe4?amJ`EypW>mst}ywwc-yGGTr3k<^i<PMcJgxAuCF4!#*=5nmBP5RP_YDt>)K0=IgxtK2*(Un^=cFOSyvKpT_>Zv6}YH#'
    'r97QiyEiO*NDe=UIr-'
    ';vvF1nPu2sG(1<>_+GGYaUMVG+wej3@b`_rKV9wF*7&~1A3ahYLjf~VQkSCR%8h$V;FT7$A(%#ARRTzfD@Xru>*IzmepHHyzT@n=oC++?}HW8O{B|lJnt8_I'
    '4Xz$ys$D3)z%esp3cchA0@Z3p3MAiR_+lzh#nBtf}Pfk4pEk9_5<GBHap575mzHx?PO_O!&GP>N7)>_Hjv$ZaNK>6&N9%UK6C}0A?%$i4;FRH*v`Tld15CYC'
    'I_S$gwgY6UweT_pL3Wxh#!SP&zJFlOSF>9$gQ?t`GRLOSJZI1f>97I+IMmCB75LpAA$y%=W_xJsgNJB2Mhd$Sj;}Ll_hrhZK99+YR4L!%vqQsN8l*apo0xqQ'
    '|7ricHmsf6f7j$Wb@>R<TcK1V4N4aUJIKv3oB`kIPfSjvJ&=t+cIxj*=9y9qtI1N7KJk+wuwAiry$AtZX)WjhcZzqo<M=$o&;G~CS&Op96O3q*fofD8=9qYJ'
    'k<#@w2q3q)eDRbXDDW1we(~2*jIWW3!odU>mJVDCH7ZA^`-^IgLZthef+-'
    '$qOV89U|!+eWmTFpygJpA2XZO*^qZsbmFim~nQ%stMzy*pz#z?Z^!wMS5j%%v=s2itK<B%+!a_JiZ+A?sn8a_d;7qm_**1z*mztwTcDgPYS#RVww-vMhkKka'
    'KzjTeIoFFn3CUbm`6`3T*_?+zDJ*==&@`gL`V2{`>jqNWHwXh<<8i;uw<&_xO;66RfDD}x7=e=423rui!&7u7Ih>h#m<O{K$U3QET-'
    'U#a#4(n6zlR1sqfyQrI_e|npI0NDZ95x2}FwH65OMG|<PMg^@k0>K{l2zDN)IYLipM<;9qp{4ORoO+%Wf0xWM!ds2m;&3JLgsunqfWM<@767;rOzjDe>Zs-'
    '9*H76;4z=0W9NM$1;=9zjT7q@7V?ChNxX<E$RIXW6VbqZtnqqu0&RoF$PI47rXJzsEyGHGg@0kJ!&v4p+<*b}8V_MuTM5Q3FqV`k+CKa2gpXk2SNIzB`qT`C'
    'B>6>E<Ur=x8?NCBIj(mE2e^d{Q%9*cu@mdztnQFeyMR*bF6{n$cy84o#jE5v9fF7_V1F09<pFjq`=;aAV&=9)@#gwb<#1&G1)MVuw<3WL<9yJ4>_neFcd%Hk'
    'q4ACmUIzBkEllle>?PY<bL^>g@Goy!-'
    'KAi(W8}WSf=Fj!Gwr}J?=7zSZM+bx6JB7U+k+}cC3unOuQ?v2aKjEzTy&H#!fxUDw<ofPn$8mY#wGeh=GJuZR+oa|l%f)Li9b80`;7kAVF_5}cmr<yR599<f'
    'U{o#7x4*2y9F@sO7axCu}bYT)}rK{BhSm+S!0UUgzDv7e^z}5#2zX*H>??8mSF?_xjmz1yzC@#S0l`)<VN}wzhZ4We%-'
    'It+lMFEPgYEfERPO!c?vqHhalzKctx$e@h8~(BQRes^@LR>W<VfLh<7{RW(fxTD)m<r<4dxWwctTjV;#(a6-'
    '3Zzn8VtxBi@iCljRv6k2x3%=$4#=6?+Yw@#b}aO18p-bk6Azwnd;{6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7X|u7fqqe-'
    'UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7X|u7fqqe-UliyU1^PvSeo>%b6zCTP`bB|$QJ`NG=obb0MS*@%pkEZ|7X|u7fqqe-UliyU1^PvS'
    'eo>%b6zCWIujm(<f#Wvrs)kXlt6}V)DA|0r<eOCM12F>Qq#Jn9O)BM?Z1gBfMhPr6dkQ&t=hZ4Bj2E>7FJPQAW`Nv<Ml7of;smQiHc#P)trK4wz>19%J2UfO'
    '2(C#)Gt)vI${46)D_T8fGL?q>f3m;A>xz=YZoZoR_aS;#$8J*w+PBUQFbaPD1gCdPPwuN^V;6TYc&pLfv2C-'
    'YKCh7#x(&Jz(U(WfUcn$dsypJ|o#YaZsz;~TcP+#`&#`CA$WC$ueiAmDy|fk7o?h^m9Zz=*ZU%(ri72ko_ZhJKMf#S5?~%ek^D}ScZ2LUg$K&i!oA$^a8X}f'
    '4!(7+H^RR_^N|swEIi2(D-4)=(W(S^QF%H17Fh{K#epNZP^b0sc3!LX1EV5Hh{0_Be4}@-'
    '$ctAZXVM%wJDdT)%v#!f{vu;(}EEZ)und&*sb_uxnZ4qnlMv29d{w>(Pjyt3@j!R^QTi>dkOuR~>tgo7dZZwK!;rHC>tA(@Gx)8IhoKAA}Qe?mt^3Cj%_G$l'
    '$kbAXFJabt0wCZY{2*e(HR6o(eQerfd=oi(1DQ47Kuca^<x=(N%M%9)rl(4LA<Wt35jRY3zE+@}ASrjQULGq{tO{00V2RorgWzJ$jm`PKoewL!6lS99#z}cM'
    '86AQ&`p$Brr)?pesDsNOU4TEV8tI9bsVNSnJ&XArD{h}%|wvNypu~x_^xKe^BVGaA3dGNN1_u!^B!)NHkx-9slF)-;Ga@dZ*7R>xO#A4Z3*_#@PY9b+ZoME-'
    '<Lk@PqJl5j8%4yP*)^pf^59Zw?52Ij?9IO;J{|!FC7Mz|@BC=LO%IKaevTMv%I3q*k4dn2KQG0-AG06VkNtVhAEHPW@&%{jcVZYaqT|18wM-'
    'pVCmRNq2toj!8i^5>}b>wEZ!Ph$^KWZO6n--M6X5m{G)HeL{4&<Xz_4<w5BTsmmGi(5M)&UH^CKQt%!Ow@(-'
    '*t`KAs%xl9P>D5!v>tDH?qfWv5TT0Th^?3MDeErUi~3C`EO(`G=LO2&$p7uwBO1H@>*L}X1&H@Mt!lLc7WwSPE@%Ey{AR=f6nnc!i@SI^Jp8o0$rFFxZ7sU6'
    '}a(l@YrgZy+txSu9>kxFrzeVs!9~yN?H5u8u5GHqZ@dQ*7<oOs(-G}9$}{03Z4fse*^b>Ag^-@&SU~q=nj9wc-'
    '#X%UIshrf*g<=m0*NsL+`L7iQDja#>fWrdu3L+857<u{Z`t}vZZhiws-'
    '}Gx<u~gklNWzhDsXL^IXr?fL$)ItBmV<n!VL5uP7OjcVt<t7qza;*_+U3T_DL(tpGD!!;Hs0w&tl;Ks7negPaQe<Ys!m^V>Pj>3pYl-'
    'O$^aqDqrFf~WX{7iw)8`XAou9C7Ufxk@nWmE>X%f;xAgVYWg<Vg>$c4vuv#Ol0FfF(Nmm#9zYL7_Bf?dtl@H#CSu$Xot+01N`&>GN2+@E?aOz`?0i-'
    'P*HHMSt%G!BRK{$@LOleMBl~sIw2o+hKS%OzTySB;+JG0=GE4DttMM)jySKNWR?Y!K7tv2hDwkloO37%ToLQ9(A^&C8F&PJnjvfOC9s8@ftXSG8`!<iy8lUE'
    'j#4$l^*ZB3=mooVhMec?miBLJ?9Kh&VV1q?MzEBORKCGEtcJbT3GO{fUR;d5uSH|y_~bVG$sCM?1JL^%aTMDP;~-'
    '&+?1p9P%R2GaMzRjP3zT9rc}JZ^Q({wN_Jq<(BMJq>SXuAP<s=B}6Kg4qm6gZZnZgd($41^`cWs8hJb?E!qi1Kx%qW3Zw5R9Gi>Q&Y`zvDd!)kd8x<3mj95@'
    '%m{?$G2{S^#u+km(FR|ccDS}ol`uWK2m;36F9J$*WgHDART72?h3AjoO#M&~Mbz(vfdo;&cQ9p1rVGVdzL0rXkXtGcc=hAre_oWVms=DPEJt6_k?6YXlzSdS'
    'Ai?Ns?$&Y)G||21U6IZLdM8Mz~tvk9J?z@v$QD_8|%4vWv~Z4vd~-r@zPWm;`0BcID!YZaUa1<as_G3>wsiJ@O~4u`lO@7y_qt-'
    '5CxE=W?Pc@7rH!Y;ugN}*r$jE<$}&<c}&?;i2#&xrv(vR<7n>ff4p-'
    '|NxaF&}aTO}Ke@V58(z*#;<N_@&tq>(!HH=E1Dd0v9rlO?rpDQi&xHA@aBp3K7vCCxdYUpTx=~1x<=O)J&tGrUR65MP#}c%+c0{^M$>Nhp^;&VQb#PSv+UII'
    'YIr&Z&DL>O|{1Nk+-giJS6LSjfN8Bp_kzqwUf7S1KY~yMjg4feoJeJYTpsvKZ9p?N7QGZxfxUa+gj%@Xpw9aL0M!y-'
    'BS8EocAQQX&ZLx2{GYG)siBfpC(#fMy%31e;d)QJ=GV6PqfX6Qw^uB8?VaosR)tVJ8%WFIags2T%es$NB)r!&tAB)Js@)n=p~x}`3y%dLloPru_|W1f*iasn'
    'cH<l*_^NYh$nJOL@8Bdv7Uex`9$x6$+4t1x~^l*>M%R`y-'
    'MCBGsUVUv&5<&>3<nK^j2)z6Yw1KQ>Msev)@^%uL#`c6EZ@htdDuJMWTAHlCxtD^e4$WJ|$=IO8t$Z4c1Mx`5t?x6O`qw$Yk`K^^xsu#<SV5EgI`)qH|GHQi'
    'gR`98C29?!qDJaGfaSo-qPD<l~w**h8Gi+2Z#qHwGuuaiS~mH?x49iDx|Xlz2<2?rMVBGEr2wr~vsNz*fi-#TWp~8h|4`h3e3*>RATCt-#B>!NRM8-*?RE-'
    '=Wr8n`0Y`&uila=65y9lMQ4jkDy<aL9w(BhPqF!d912QaH3aGcIWV(keN0{Jh~3$19NfHtfN|(We4n(ZRE-'
    '2(PyfH+5Lupmm>otqWdq1Umbyom{*%etIqFZKdWXEguyw;TPUcufn!@kq>uc2=T-WxAI59TVxv3XyNOI4tKpsyzkG)^wx^buH#JU%-'
    'Map@Q;VbV1$_OEUEAn+!H(D<8#ks=tblD&h27(<n0BJHEyPQk*&U*=rhGm+7cQ;ZW_V*uu`ypjIS#?L&FAphFpKxO&q;L(C&>Cm)#QgC!DsFwdKyNJWdN+Do'
    '~-W|_Lnm9iMlDTmhqVYKR8A?cMvRRi%h*?vN^9l^ou%Rz%{YDtSy>E54A<5PNOI2tPJZH-7#C{Yt-ON*hY9DkIC8~2dH1B@#@?8zDeY$84GkARfBTaJ9R`K>'
    'fxV{V3pLNI%&MA4Bxh$^;J)-p|H=?5L<4;?#!`nX245!(fnw`HW?s>Y@NtTG~?n${UW2}Rxey6UNuiX-'
    '4<D=Jw^SZS=dWwWC;%AVUB_QbfGnuQ@cvZV)pr3%bMzEH&_eJ;9rcwswg0zeV{K-tPdlz=lCAZp%}&M-'
    '6l3^hSmWYfxBelFOzA}haUDR=x__8yh|o<HMt@~MX#m?E~J%a4#D^;h%Mg1n7U)F?w}N8MGa@Xtb+(%;r}e5=}`(h|AIH=lq}XEY>z$FXN<0&Od~UkSF}ful'
    'a=9&n?V?VEo7!2!nNCDH@Q>$?Y$k*yFJ+9J*W%B;a9xT|5><^rNqx`)!IYi&{x>WWyBoL;ZMC1^*)3(Uk5K{24uUDytr<)Iu2)Nk(|g8Y$3<f9Vgy^<(ELeC'
    '`@d^I<OIV)aLBmv6^1+NUVs~0mJGQj$Au7X*UtR8GOxN)~;0zjYhq3dXDM6FiUR&{h|o@6KzD!3rfZ9AZ1>UQB+Jfv1!uSrS^czS&>cZPaYe?SV$jlgDu8w_'
    '7U4X`|48y<*rfQk!|%W%1(7lrbvXAKCx%}ZL%F{eWhmnk!AH^h&-bQ(1c-h<=RnovA#<it2qyEp<pfCa(YI=<nGAlpN0W<q#k-->nUQz&PFf}-'
    'AC_NQCc}dF)gL1mjeBwK))!^FADUF0{x;uzbMcz3iOKt{h~m>D9|qo^os)hqCmeW&@T$~ivs<kK))!^FADUF0{x;uzbMcz3iOKt{h~m>D9|qo^os)hqCmeW&'
    '@T$~ivs<kK))!^FADUF0{x;uzbMcz3iOKt{h~m>D9|qo^os)hqCmeW&@T$~ivs<kK)>jJNWaL@hh}hz6C$jA;2Tl$;)cPyoWXBR7DxLk(Y;7>wTH+}7B91%J'
    'hCoU_#sU97LDR5`E;vr35?X#aKb$6S_Xb<1B~EvwepC4Zi1ZF3$=NMc{^7bjbsDwU^|wQ?N>rpZ9mNLaqQ9pZ~d$of4U|^q?En7fk<E(9AcmL`|9s2nM1>5-'
    'FLu=e}H>!%d3%yz$%eUBdD$9Cyjwum%`CK(&$+4VP37cBpQ}QU*>{UZlwR1JnS5AvvazKh>1BP!0$|gJ~zP<PpUpE5VaF0GxK8(exR{z^F7<NhI@(6I%=>Dd'
    'Ogp1YTVIYLmS!QgRD3s`d#>OId*HSx|G3g^a)gfS7bg$Cy}CV@WpQ4JS!}u;H593A&}wpHg_lwcD+E3ZI%p4^P_f&=|w=XM)1b+p#9UjyGvIK`bEwHSpkE5A'
    'lqu2czcQZ@BM!VtbH7>wToTPjOJ2(mLY!Zh|D`keVm=_l=w>qW~?K?@2br)VCz(Eqao#JZ4S@#9Dm(e5*y$U#&}!UA2zW_jUdgDx9GUCwTW8s8VdY}0l0$WM'
    'CZJ-W<ZsFPfJn1Xo>xP6)u~3kR581tqE&f)xc$&RIR08F$Ki+1%7HZNPHLiMH!9Nyn6J=e%B7?^92uJQRSPfa-'
    '@3qh&NrU^lM`M){z)zms{pkZNMA0MN|&%Y7Lqbyq*cLniT%J)o;f2c`1=><4O-GotO(bthRQ-'
    '#M#3>a4wH?y_)se6Q_C%Ms7jBXk9I6#CrCwa18^+`bCbaO@V_ZiDBnBndjMa3fX`5wygTn4mWL#2;>}l&?_7kGb*B>XjiC~?7+vd&D>0kbq0Loj+4%;vkVzD'
    'HCU_F_(aZEcecp_h*J~n23r>w@CFNB>u`~0_X-;&&zgFGeetAgPrT=K_)KNQy`tm_Ke0--K${Lhtam{BpU5$9Cqi$_!nSIP^Y>lzGmmP~EME2`5z0Ia_-'
    'D}g6I88Ei5%_exmFmXQ)r`1^QM;+&!IBB+9(Q08}#9j6Zj3=p#knu54_+WyqG&q@^dsbF2KI5;LxuYUGcv3!RdQK0bv6s;{>N>4ix}r)jEb(uQoWMnj;snN^'
    'P9ww@KJYW}^0^5q7O-JIJ@LDC!r*$h*F0El-'
    'o(xm(mPS|yX!9N;R{m?Ge<=0>a#2YuuHw0=`k<M@bIvjls*mH1X+=gy&vGDJRHyFTCH*_7t9m36TSf4u_zqB`uAEm-8vTK5{VHE2s=9vk@q6r-'
    'M4_7;p&$8BC<+D^a{F}Ai&UW^qgQYgrHf2x5EW`D`OttB^q0(On#=~n)+*WqVk#9LDuts2mXW$gKPFuzr>(6sutMwIf2oG>$jvT$MA*qMwZzk@$agGHs$(Yi'
    '(H$oEIkq<PfHb;00{!(BG!b*SeusOTi&c=mu(W!0KZIJUQ79$vp$SatVAeVwKLLL_3AH|`v)b{jk44Bo&z8IDgxpVQ#dj@s3M`m|v^n%x-'
    'V(+W<+VUR=XwymPRHi{06Pm3(<h9@FZ^JryE!y!(Qy_O|IYZH6?fz!T`98+hsT@lBy2j;x2o}zaVN59C~B+sB{U9jgIzwvr_0<~@<)8R=i3#*nPGE_=A|HEj'
    'YI0|+{?N2cBqO73<SXcJj-astd$W&e@)--{|^GXC^Tys^(1u0W|-'
    'x#??qOaBfih<9ck#SqlFPc{G+Vyu_zgNJQz2;0Ah7mf1Mn;OA<Q|p4cJ<R1);lQUDlsbO&pcqmm$5&aE7?U%a6v6t(Ky_bgY21gw&ejT7%fHpBIh(*5^3>W>'
    'sV7}5e}mha=~f{Vf)X+5Zb~9y#mu7gws;U-MvGXCkw04{K7qy4$4pf7=<nOz`C`{og+u)`#Y+>POkF1S~beM>8N5De#0SI5JsQ3=(G9r%b<l1<m8#t*@4w%-'
    '4&x~*0~$hT+E?dW9(}cAG@m<`>#T+#cCb(c;Bmd=&PvgO|r)%@HMTK;xpugT>W+QRff?=n$UIcF!%bzt!dZy3i8+-'
    '=kt59g4bd1Wft6P6x8sHtm`<tNFS{5GGa5%19*gwRM0OP#g=VDzsOOAe)u8fFm?<2MPuk~^@6uIVX1ehgc$jv{lt1ZVdzh*gbK3wo``-e!-'
    'rgE@2Lm<?89Q2WoI?BJ`WDMNxuCVnZ7sV)b*e$HcD>x8qD1`)wl&~?izKJQu5DYWWMgAX*7-'
    '6Rf1FXniIi1?_SmJD7RHrtirIzUcO2!?Uuc5pLgz#$Urx2fr5rg3;ESo<ZNs*dL=~n=TMVKgM6>zdzNY(D&c46$k<s1If#LByFD*NlMDJq<9f<@1<v}g&SeS'
    'y^*me@$8C&AUZI}!L?pZ(RI`c*M2?ts2}%=T@*wQjM2H~SV;Tj=zCj~1NuJIM2&=OlPB_=hjF<#dnt&f+zDhMw{Ul5w`>>1T%pS7qK7$bSsqR6dFUMHD&W`N'
    'Tc%(svZ(%?^!4Py_^9t$>X1R2t*l|eq$1$7~TYSBsB?nlc)-QS^OSp}*EK2s;J4}m1R^~G(XEV(EG@QnQevxBrQPf+8;9!?xk%q}9aTe5(Mtg-fd4U+A<1<f'
    'K9*+O{4X_n-2ln)g!jU-'
    'v%POmu4DT!6G%M<^74?hGU{<f819L(gp_N?RR=jHKQJS@Xh&tOASsvDoOo5l#?zkXlqZu5<5rBUE9Opf?DpCe*gA{mTM)l8QqqlQ<G!lbahEX^{4&M}5r`0c'
    '-(A64YFWmum-@ya02m4FE$UG)TDcU$IX2>uoS9xZ4M#zFlkx{){RBrsGUo?ZZYBj984mdMoFh}-b9c-'
    'hpXAP?a_(>DYvRmTY{ix+$^WI#76lKt%@#uwlD=!$UcNj;tD#=lYN6qIXd5$sk?VfaxZShH1OShmvE;Wf_Og#uhl3a}f3#S1Dx|Phmf_~9C@yr?Q8E1}d!A>'
    '1U8RH5~s5174I2ung<j3@rQPGb!L^ayQ74TjgVK#KZ&2pS(fHmKYznviV?z6#>p&N9UXNZ`Y!`?u2$@&mYD66c&8k`~0xyrpI@RAqVkB4BUCshAB`*A0{b)!'
    '0qtbqvm<3m^$)ns0`qlGa{UXvLilPYfj-'
    '=PzR`xYF5Gq_yqFhfQ;KT1%ajZ)_`xMUa673(9O;oD!3GyjN|<0IL|L#m~dSo;o2D0lF*>d`WFzF!y>%0_hC28gj*YpFx`jDiC!z?^DSZ%V+QUx|-'
    '3z%ELl<8Y(B!s?y3c*oc53Z*dHtMoQoo4Ox$j8RG}SIf#(egio2G$-~I>L_o_e3)D)>%kph1N37ln5z-z9e-'
    'm_Gur29b_J+`bN{VpeW+eMqW)@h(jNOL8W6|o;WD_1{pCJzDre(IVML9qJttsFFJ!dckW+p@T=i0OF$8b^24vCPU$c75kSHgQy97^Q7v88<A|slaYn5XS+XJ'
    '|L-Nc|B<4vK)>Fj-D^dl-'
    '~nv5VL`sUvyz&&<}P3@rl*2K7ZU$afWOWid@OrF8`>#;b!tIu%%F=y!lTqBR9cgitzC44Hhfk4}qboO$X9i|rMLNB}gCDxx2f^yKH4Vbnuy|d=H_Tm9mvrZ1'
    'xw^OXwH9VaS-m`7AgdAPI1<|pEXumv<P4U3FUrqL^6?6N+F1yvH3UHHim@`#m(07on_>3C$4B7f+FmEov+p=IKy+n(1VC8e<%nXz5d5XPLfmLgb+7-'
    '=_^JjX=to3f2BtP9JyQ9!%fFI)#yr4`yhgz7K@>66tM95V(N5_c0nH*ba;W=+UNj^X?T<Bg-Y-bmH@3rl_MYdBKlw}$%f>YkQWqjE<=;s62%Q`taO;~s3=oX'
    '$4*)Hf8^%E&LCZ}_OoP-'
    '^4fMXCet3cjs6h^Rh+VN)NM0R)Zfp*ajaaR3p=>G#yO9KQg000080000X05I@O{{R2~|NsC0{}=!i0B>?>XK8L>Y+rP8VQFq(V{dhCbS`dic>w?r0Ehqp000'
    '0007n1-000000GsR=>KhQr$WX>mt)7xvoLr=CrJ!z;X`!y8pq`drR8o|f7oT60k_r-'
    'cOUx-w1&SAEBo?Fs`5Kx!3bhIp0j@Gh1~32sP)h*<EdVV5000O8001EXda5L&V?6)>KuG`q3;+NC0000000000fB^si003`rWpZI|WOFWVaCuNm0Rk-'
    'mEdT%j2mk;8ApqL{^%`m^002Nq000UA0000000000004jip*;WqbYXI5WpplXaCuNm0Rk-'
    'mEdT%j2mk;8AplfhBw=)30002B0RRpF0000000000004jiM`r*4V_|M&X=Gt^WpgfWaCuNm0Rk-'
    'mEdT%j2mk;8ApkJ&O#Vjz004*p000;O0000000000004ji`^5kNZ*pm8X>MU`UvzR|X>MO*Z*^{TE^csnP)h{{000001ONm8^Z)<=jK%-}000'
)
PANEL = json.loads('{\n  "protocol": "mod97_frozen_grokking_function_duels_v1",\n  "source": "E:\\\\Downloads\\\\results_power_mod97_grokking_mechanism_n70_package.zip",\n  "source_sha256": "fd4adb639797d48869280d580b4ba0de0c674af70603b29f14995d65d7752802",\n  "payload_sha256": "456abea32ea48250b58a0cb4cd01b26cb1ace38d18fd956b3e5aec3bb1773d38",\n  "original_config": {\n    "prime": 97,\n    "train_fraction": 0.7,\n    "data_seed": 0,\n    "model_seeds": [\n      0,\n      1,\n      2\n    ],\n    "conditions": [\n      [\n        "ce_stable",\n        "stable",\n        0.0,\n        0.0\n      ],\n      [\n        "gap_var_0p1",\n        "gap_var",\n        0.1,\n        0.0\n      ],\n      [\n        "gap_var_1",\n        "gap_var",\n        1.0,\n        0.0\n      ]\n    ],\n    "d_model": 128,\n    "n_layers": 2,\n    "n_heads": 4,\n    "max_context_len": 50,\n    "learning_rate": 0.001,\n    "warmup_steps": 10,\n    "adam_betas": [\n      0.9,\n      0.98\n    ],\n    "adam_eps": 1e-08,\n    "batch_order_seed": 20260901,\n    "max_steps": 1000000,\n    "device": "cuda:0",\n    "allow_tf32": false,\n    "model_chunk_size": 16,\n    "eval_batch_size": 256,\n    "eval_interval_early": 100,\n    "eval_interval_middle": 500,\n    "eval_interval_late": 5000,\n    "log_interval": 1000,\n    "checkpoint_interval": 5000,\n    "stop_at_test_perfect": true,\n    "save_first_train_fit": true,\n    "baseline_condition": "ce_stable",\n    "delay_low_test_max": 0.5,\n    "delay_min_plateau_steps": 1000,\n    "delay_min_ratio": 5.0,\n    "result_dir": "/root/autodl-tmp/results_power_mod97_grokking_mechanism_n70",\n    "resume": true,\n    "package_results": true,\n    "smoke_test": false\n  },\n  "source_description": "Power原版455424参数Transformer架构；70%数据，CE+0.1幅度方差正则；不是原头图50%无WD严格复现。",\n  "input_definition": "完整9312个模97除法输入；操作数用两个独立one-hot位置表示，不增加数值大小特征。",\n  "output_definition": "保持原239-token词表的答案token映射；不测EOS，不删去候选的异常答案token。",\n  "selection": "来源模型3/4/5均实际达到全规则正确。near：完成前最后一个训练全对且测试<=90%的记录；far：完成前训练全对且测试<=50%中原始答题CE最高的记录。对决前冻结，不按胜负选择。",\n  "source_final": [\n    {\n      "source_model": 3,\n      "step": 140000,\n      "source_ce": 7.620029297630334e-05\n    },\n    {\n      "source_model": 4,\n      "step": 51000,\n      "source_ce": 8.492187594864688e-05\n    },\n    {\n      "source_model": 5,\n      "step": 46000,\n      "source_ce": 9.967820225836501e-05\n    }\n  ],\n  "cases": [\n    {\n      "source_model": 3,\n      "source_seed": 0,\n      "tier": "near",\n      "source_step": 61000,\n      "source_ce": 0.0003838089621490745,\n      "source_test_accuracy": 0.8693629205440229,\n      "source_train_accuracy": 1.0,\n      "original_hash": "5fafc910d130e00a8b8872effaeb1d2bb4c15a5999e85b546a0d58517bd7fd96",\n      "source_prediction_file": "predictions/step_0061000.npz",\n      "agreement_count": 8947,\n      "conflict_count": 365,\n      "final_source_ce": 7.620029297630334e-05,\n      "source_ce_ratio": 5.036843654504463,\n      "case_id": 0,\n      "fallback_case_id": 3\n    },\n    {\n      "source_model": 4,\n      "source_seed": 1,\n      "tier": "near",\n      "source_step": 34500,\n      "source_ce": 0.00048324913829482583,\n      "source_test_accuracy": 0.8793843951324266,\n      "source_train_accuracy": 1.0,\n      "original_hash": "1de8697f685ad5857039e6ec17c4b43febaa96551f6044c027dcb592b0834480",\n      "source_prediction_file": "predictions/step_0034500.npz",\n      "agreement_count": 8975,\n      "conflict_count": 337,\n      "final_source_ce": 8.492187594864688e-05,\n      "source_ce_ratio": 5.690514168422887,\n      "case_id": 1,\n      "fallback_case_id": 4\n    },\n    {\n      "source_model": 5,\n      "source_seed": 2,\n      "tier": "near",\n      "source_step": 28000,\n      "source_ce": 0.002214577404207401,\n      "source_test_accuracy": 0.8965640658554044,\n      "source_train_accuracy": 1.0,\n      "original_hash": "5a318ac01db728b26c1f60a9e125f98bec561795650d73132c2266e7544aad57",\n      "source_prediction_file": "predictions/step_0028000.npz",\n      "agreement_count": 9023,\n      "conflict_count": 289,\n      "final_source_ce": 9.967820225836501e-05,\n      "source_ce_ratio": 22.217268711039114,\n      "case_id": 2,\n      "fallback_case_id": 5\n    },\n    {\n      "source_model": 3,\n      "source_seed": 0,\n      "tier": "far",\n      "source_step": 1100,\n      "source_ce": 0.12419920445627114,\n      "source_test_accuracy": 0.022548317823908426,\n      "source_train_accuracy": 1.0,\n      "original_hash": "1c7f6026c6e332d0205df11b0e14828217efb42a829d9d053b403c323191f34b",\n      "source_prediction_file": "predictions/step_0001100.npz",\n      "agreement_count": 6581,\n      "conflict_count": 2731,\n      "final_source_ce": 7.620029297630334e-05,\n      "source_ce_ratio": 1629.9045529246775,\n      "case_id": 3,\n      "fallback_case_id": null\n    },\n    {\n      "source_model": 4,\n      "source_seed": 1,\n      "tier": "far",\n      "source_step": 1100,\n      "source_ce": 0.1106844145189719,\n      "source_test_accuracy": 0.023622047244094446,\n      "source_train_accuracy": 1.0,\n      "original_hash": "01233f514a34bf5d67ac9fa0a9a06bda4d4663fb3d583948f0def66a5aecafdd",\n      "source_prediction_file": "predictions/step_0001100.npz",\n      "agreement_count": 6584,\n      "conflict_count": 2728,\n      "final_source_ce": 8.492187594864688e-05,\n      "source_ce_ratio": 1303.3675161145038,\n      "case_id": 4,\n      "fallback_case_id": null\n    },\n    {\n      "source_model": 5,\n      "source_seed": 2,\n      "tier": "far",\n      "source_step": 1100,\n      "source_ce": 0.11491750857928251,\n      "source_test_accuracy": 0.028990694345025103,\n      "source_train_accuracy": 1.0,\n      "original_hash": "0b13b59697aa7a470838be96f1fe8f7f1a05e7b4171ccd26b4042870044b3905",\n      "source_prediction_file": "predictions/step_0001100.npz",\n      "agreement_count": 6599,\n      "conflict_count": 2713,\n      "final_source_ce": 9.967820225836501e-05,\n      "source_ce_ratio": 1152.8850438275097,\n      "case_id": 5,\n      "fallback_case_id": null\n    }\n  ]\n}')
STOP=False
LOG_PATH=None
def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return ready(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def replace_retry(source, destination):
    for i in range(8):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if i == 7:
                raise
            time.sleep(.1*(i+1))


def atomic_text(path, text):
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    replace_retry(tmp, path)


def atomic_json(path, data):
    atomic_text(path, json.dumps(ready(data), ensure_ascii=False, indent=2, allow_nan=False))


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    replace_retry(tmp, path)


def write_csv(path, rows):
    f = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(ready(v), ensure_ascii=False) if isinstance(v, (list, dict)) else ready(v)
                             for k, v in row.items()})
    atomic_text(path, f.getvalue())


def log(message):
    text = time.strftime("%Y-%m-%d %H:%M:%S")+" | "+message
    print(text, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text+"\n")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def atomic_checkpoint(path, payload):
    tmp = path.with_name(path.name+".tmp")
    torch.save(cpu_tree(payload), tmp)
    replace_retry(tmp, path)


def fingerprint(functions):
    def encode(v):
        if isinstance(v, types.CodeType):
            return [v.co_code.hex(), v.co_names, v.co_varnames, [encode(x) for x in v.co_consts]]
        return repr(v)
    return hashlib.sha256(json.dumps([(f.__name__, encode(getattr(f, "__wrapped__", f).__code__)) for f in functions]).encode()).hexdigest()

@dataclass(frozen=True)
class ParameterBundle:
    hidden_weights: tuple[torch.Tensor, ...]
    hidden_biases: tuple[torch.Tensor, ...]
    norm_weights: tuple[torch.Tensor, ...]
    norm_biases: tuple[torch.Tensor, ...]
    output_weight: torch.Tensor
    output_bias: torch.Tensor


class BatchedIndependentMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layer_count: int,
        model_seeds: Sequence[int],
        activation: str,
        layernorm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_layer_count = hidden_layer_count
        self.activation_name = activation
        self.layernorm_eps = layernorm_eps
        dimensions = [input_size]
        dimensions.extend([hidden_size] * hidden_layer_count)
        dimensions.append(output_size)

        all_hidden_weights: list[list[torch.Tensor]] = []
        all_hidden_biases: list[list[torch.Tensor]] = []
        all_norm_weights: list[list[torch.Tensor]] = []
        all_norm_biases: list[list[torch.Tensor]] = []
        all_output_weights: list[torch.Tensor] = []
        all_output_biases: list[torch.Tensor] = []

        for seed in model_seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                hidden_weights = []
                hidden_biases = []
                norm_weights = []
                norm_biases = []
                for layer_index in range(hidden_layer_count):
                    linear = nn.Linear(
                        dimensions[layer_index], dimensions[layer_index + 1]
                    )
                    norm = nn.LayerNorm(
                        dimensions[layer_index + 1], eps=layernorm_eps
                    )
                    hidden_weights.append(linear.weight.detach().clone())
                    hidden_biases.append(linear.bias.detach().clone())
                    norm_weights.append(norm.weight.detach().clone())
                    norm_biases.append(norm.bias.detach().clone())
                output = nn.Linear(dimensions[-2], dimensions[-1])
                all_hidden_weights.append(hidden_weights)
                all_hidden_biases.append(hidden_biases)
                all_norm_weights.append(norm_weights)
                all_norm_biases.append(norm_biases)
                all_output_weights.append(output.weight.detach().clone())
                all_output_biases.append(output.bias.detach().clone())

        self.hidden_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_hidden_weights
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.hidden_biases = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_hidden_biases
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.norm_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_norm_weights
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.norm_biases = nn.ParameterList([
            nn.Parameter(torch.stack([
                model[layer_index] for model in all_norm_biases
            ]))
            for layer_index in range(hidden_layer_count)
        ])
        self.output_weight = nn.Parameter(torch.stack(all_output_weights))
        self.output_bias = nn.Parameter(torch.stack(all_output_biases))

    def full_bundle(self) -> ParameterBundle:
        return ParameterBundle(
            hidden_weights=tuple(self.hidden_weights),
            hidden_biases=tuple(self.hidden_biases),
            norm_weights=tuple(self.norm_weights),
            norm_biases=tuple(self.norm_biases),
            output_weight=self.output_weight,
            output_bias=self.output_bias,
        )

    def selected_bundle(self, indices: torch.Tensor) -> ParameterBundle:
        return ParameterBundle(
            hidden_weights=tuple(
                value.index_select(0, indices) for value in self.hidden_weights
            ),
            hidden_biases=tuple(
                value.index_select(0, indices) for value in self.hidden_biases
            ),
            norm_weights=tuple(
                value.index_select(0, indices) for value in self.norm_weights
            ),
            norm_biases=tuple(
                value.index_select(0, indices) for value in self.norm_biases
            ),
            output_weight=self.output_weight.index_select(0, indices),
            output_bias=self.output_bias.index_select(0, indices),
        )

    def activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(values)
        if self.activation_name == "relu":
            return F.relu(values)
        if self.activation_name == "tanh":
            return torch.tanh(values)
        raise ValueError(f"未知激活函数：{self.activation_name}")

    def forward_with_bundle(
        self, inputs: torch.Tensor, bundle: ParameterBundle
    ) -> torch.Tensor:
        hidden = inputs
        for layer_index in range(self.hidden_layer_count):
            hidden = torch.bmm(
                hidden, bundle.hidden_weights[layer_index].transpose(1, 2)
            )
            hidden = hidden + bundle.hidden_biases[layer_index][:, None, :]
            hidden = self.activate(hidden)
            mean = hidden.mean(dim=-1, keepdim=True)
            variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
            hidden = (hidden - mean) * torch.rsqrt(
                variance + self.layernorm_eps
            )
            hidden = (
                hidden * bundle.norm_weights[layer_index][:, None, :]
                + bundle.norm_biases[layer_index][:, None, :]
            )
        return (
            torch.bmm(hidden, bundle.output_weight.transpose(1, 2))
            + bundle.output_bias[:, None, :]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_bundle(inputs, self.full_bundle())


def stable_multiclass_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """逐样本多类 CE，避免 true_probability - 1 的消减。"""
    target_logits = logits.gather(-1, targets.unsqueeze(-1))
    relative = logits - target_logits
    relative = relative.scatter(
        -1,
        targets.unsqueeze(-1),
        torch.full_like(target_logits, -torch.inf),
    )
    log_other_exp_sum = torch.logsumexp(relative, dim=-1)
    return F.softplus(log_other_exp_sum)



def load_functions():
    raw = base64.b85decode(FUNCTION_B85)
    if hashlib.sha256(raw).hexdigest() != PANEL["payload_sha256"]:
        raise ValueError("内嵌完整映射校验失败")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    assert data["operands"].shape == (9312, 2) and data["candidates"].shape == (6, 9312)
    assert np.all(data["candidates"][:, :6518] == data["target"][None, :6518])
    return data


def training_indices(data, cid, dataset):
    equal = np.flatnonzero(data["target"] == data["candidates"][cid])
    if len(equal) < Config.TRAIN_COUNT:
        raise ValueError("一致样本不足，不能改变样本数后混作同一比较")
    rng = np.random.default_rng(Config.DATA_SEED + dataset*100003 + cid*1000003)
    return np.sort(rng.choice(equal, Config.TRAIN_COUNT, replace=False)).astype(np.int64)


def science_signature():
    keys = ("WIDTH", "DEPTH", "ACTIVATION", "TRAIN_COUNT", "DATASET_REPEATS", "SEEDS_PER_DATASET",
            "CAPACITY_SEEDS", "CAPACITY_CE", "BATCH_SIZE", "EVAL_BATCH_SIZE", "EVAL_EVERY", "CHECKPOINT_EVERY",
            "LEARNING_RATE", "ADAM_BETAS", "DATA_SEED", "MODEL_SEED", "BATCH_SEED", "CUDA_GRAPH", "SNAPSHOT_STEPS")
    return ready(dict(protocol="mod97_frozen_function_duels_v1", torch=str(torch.__version__),
        source_functions=PANEL["payload_sha256"], dtype="float32", tf32=False, vocabulary=239,
        config={k: getattr(Config, k) for k in keys},
        code=fingerprint((BatchedIndependentMLP.forward_with_bundle, stable_multiclass_cross_entropy,
                          training_indices, make_optimizer, prepare_step, evaluate))))


def stop_handler(sig, frame):
    global STOP
    if STOP:
        raise KeyboardInterrupt
    STOP = True
    print("已请求保存停止，请等待当前训练步/评估结束。", flush=True)


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, betas=Config.ADAM_BETAS,
                            eps=1e-8, weight_decay=0., capturable=True, foreach=False)


def prepare_step(model, optimizer, xb, yb, saved=None):
    def step():
        optimizer.zero_grad(set_to_none=True)
        stable_multiclass_cross_entropy(model(xb), yb).mean(-1).sum().backward()
        optimizer.step()
    if not Config.CUDA_GRAPH:
        if saved is not None:
            optimizer.load_state_dict(saved)
        return step
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stream = torch.cuda.Stream(device=xb.device)
    stream.wait_stream(torch.cuda.current_stream(xb.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream(xb.device).wait_stream(stream)
    optimizer.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    model.load_state_dict(before)
    with torch.no_grad():
        for i, parameter in enumerate(model.parameters()):
            for key, value in optimizer.state[parameter].items():
                if saved is None:
                    value.zero_()
                else:
                    value.copy_(saved["state"][i][key].to(value.device))
    torch.cuda.synchronize(xb.device)
    return graph.replay


@torch.no_grad()
def evaluate(model, x, label, data, cid, indices, state, folder, mode):
    count = model.output_bias.shape[0]
    predictions, loss_parts, margin_parts = [], [], []
    bundle = model.full_bundle()
    for start in range(0, len(x), Config.EVAL_BATCH_SIZE):
        stop = min(len(x), start + Config.EVAL_BATCH_SIZE)
        logits = model.forward_with_bundle(x[None, start:stop].expand(count, -1, -1), bundle)
        values = stable_multiclass_cross_entropy(logits, label[None, start:stop].expand(count, -1))
        predictions.append(logits.argmax(-1).cpu().numpy().astype(np.uint16))
        loss_parts.append(values.cpu().numpy())
        top = logits.topk(2, dim=-1).values
        margin_parts.append((top[..., 0]-top[..., 1]).cpu().numpy())
    pred = np.concatenate(predictions, axis=1)
    loss = np.concatenate(loss_parts, axis=1)
    margins = np.concatenate(margin_parts, axis=1)
    true = data["target"]
    candidate = data["candidates"][cid] if cid >= 0 else true
    conflict = true != candidate
    labels = true if mode == "duel" or cid < 0 else candidate
    rows = []
    for i in range(count):
        train_correct = int((pred[i, indices] == labels[indices]).sum())
        fit = train_correct == len(indices)
        if fit and state["first_fit"][i] < 0:
            state["first_fit"][i] = state["step"]
        own_errors = int((pred[i] != labels).sum())
        passed = own_errors == 0 and float(loss[i].mean()) <= Config.CAPACITY_CE
        if mode == "capacity" and passed and not state["capacity_passed"][i]:
            state["capacity_passed"][i] = True
            atomic_checkpoint(folder/f"capacity_proof_seed{i:03d}.pt", dict(step=state["step"], cid=cid,
                source_functions=PANEL["payload_sha256"], config=dict(width=Config.WIDTH, depth=Config.DEPTH),
                model={k: v[i:i+1] for k, v in model.state_dict().items()}))
        a = float((pred[i, conflict] == true[conflict]).mean()) if conflict.any() else None
        b = float((pred[i, conflict] == candidate[conflict]).mean()) if conflict.any() else None
        rows.append(dict(step=state["step"], elapsed_s=state["elapsed_s"], seed=i, case_id=cid,
            train_accuracy=train_correct/len(indices), train_ce=float(loss[i, indices].mean()),
            full_A_errors=int((pred[i] != true).sum()), full_B_errors=int((pred[i] != candidate).sum()),
            own_full_errors=own_errors, full_A_exact=bool(np.array_equal(pred[i], true)),
            full_B_exact=bool(np.array_equal(pred[i], candidate)), conflict_A=a, conflict_B=b,
            conflict_other=1-a-b if a is not None else None,
            conflict_score=a-b if a is not None else None,
            function_sha256=hashlib.sha256(pred[i].tobytes()).hexdigest(),
            first_fit_step=state["first_fit"][i]))
    state["history"].extend(rows)
    atomic_npz(folder/"predictions"/f"step_{state['step']:07d}.npz", predictions=pred,
               own_ce=loss.astype(np.float32), top_margin=margins.astype(np.float32), step=np.array(state["step"]))
    write_csv(folder/"trajectory.csv", state["history"])
    return rows


def plot_run(folder, state, mode):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = state["history"]
    steps = sorted({r["step"] for r in rows})
    groups = [[r for r in rows if r["step"] == step] for step in steps]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(steps, [np.mean([r["train_accuracy"] for r in g]) for g in groups], label="train accuracy")
    if mode == "duel":
        for key, title in (("conflict_A", "final-rule A"), ("conflict_B", "earlier-function B"), ("conflict_other", "other")):
            axes[0].plot(steps, [np.mean([r[key] for r in g]) for g in groups], label=title)
        for key in ("full_A_exact", "full_B_exact"):
            axes[2].plot(steps, [np.mean([r[key] for r in g]) for g in groups], label=key)
    else:
        axes[2].plot(steps, [np.mean([r["own_full_errors"] for r in g]) for g in groups], label="full mapping errors")
    axes[1].plot(steps, [np.median([r["train_ce"] for r in g]) for g in groups], label="median raw CE")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("steps")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(mode+" | "+folder.name)
    fig.savefig(folder/"curves.png", dpi=150)
    plt.close(fig)


def run(folder, mode, cid, dataset, data, device, signature, deadline):
    global STOP
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"predictions").mkdir(exist_ok=True)
    count = Config.CAPACITY_SEEDS if mode == "capacity" else Config.SEEDS_PER_DATASET
    indices = np.arange(9312) if mode == "capacity" else training_indices(data, cid, dataset)
    identity = dict(science=signature, mode=mode, case_id=cid, dataset=dataset,
                    train_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    checkpoint = folder/"checkpoint.pt"
    old = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
    if old is not None and old["identity"] != identity:
        raise ValueError("检查点科学配置不匹配")
    limit = Config.CAPACITY_STEPS if mode == "capacity" else Config.MAX_STEPS
    if old is not None and ((mode == "capacity" and any(old["state"]["capacity_passed"])) or old["state"]["step"] >= limit):
        log(f"SKIP {mode} case{cid} dataset{dataset}")
        return json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    labels = data["candidates"][cid] if mode == "capacity" and cid >= 0 else data["target"]
    if mode == "duel":
        np.testing.assert_array_equal(labels[indices], data["candidates"][cid, indices])
    atomic_npz(folder/"dataset.npz", train_indices=indices, own_targets=labels)
    atomic_json(folder/"identity.json", identity)
    model = BatchedIndependentMLP(194, 239, Config.WIDTH, Config.DEPTH,
        [Config.MODEL_SEED + i + max(dataset, 0)*100003 for i in range(count)], Config.ACTIVATION, 1e-5).to(device)
    optimizer = make_optimizer(model)
    generator = torch.Generator(device=device).manual_seed(Config.BATCH_SEED + max(dataset, 0)*100003)
    state = dict(step=0, elapsed_s=0., history=[], first_fit=[-1]*count, capacity_passed=[False]*count)
    if old is not None:
        model.load_state_dict(old["model"])
        generator.set_state(old["rng"])
        state = old["state"]
    x = F.one_hot(torch.tensor(data["operands"].astype(np.int64), device=device), 97).flatten(1).float()
    label = torch.tensor(labels.astype(np.int64), device=device)
    choices = torch.tensor(indices, device=device)
    xb = x[choices[:Config.BATCH_SIZE]][None].expand(count, -1, -1).clone()
    yb = label[choices[:Config.BATCH_SIZE]][None].expand(count, -1).clone()
    step_function = prepare_step(model, optimizer, xb, yb, old["optimizer"] if old is not None else None)
    base, start = state["elapsed_s"], time.perf_counter()
    def save():
        atomic_checkpoint(checkpoint, dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         rng=generator.get_state(), state=state))
    last = state["history"][-count:] if state["history"] else evaluate(model, x, label, data, cid, indices, state, folder, mode)
    save()
    log(f"START {mode} case{cid} dataset{dataset}: {count}seed, {sum(p.numel() for p in model.parameters())//count}参数/模型, step={state['step']}")
    status = "budget_complete"
    try:
        for number in range(state["step"]+1, limit+1):
            if STOP or time.perf_counter() >= deadline or (folder.parents[1]/"STOP").exists():
                status = "paused"
                break
            if time.perf_counter()-start >= Config.SECONDS_PER_RUN:
                status = "time_budget"
                break
            batch = choices[torch.randint(len(choices), (count, Config.BATCH_SIZE), device=device, generator=generator)]
            xb.copy_(x[batch])
            yb.copy_(label[batch])
            step_function()
            state["step"] = number
            state["elapsed_s"] = base+time.perf_counter()-start
            should_eval = number in (1, 10, 50, 100, 200, 500) or number % Config.EVAL_EVERY == 0 or number == limit
            if should_eval:
                last = evaluate(model, x, label, data, cid, indices, state, folder, mode)
                log(f"{mode} c{cid} d{dataset} step={number}: train={np.mean([r['train_accuracy'] for r in last]):.2%} "
                    f"CE={np.median([r['train_ce'] for r in last]):.3g} fullA={sum(r['full_A_exact'] for r in last)}/{count} "
                    f"conflictA/B={np.mean([r['conflict_A'] or 0 for r in last]):.1%}/{np.mean([r['conflict_B'] or 0 for r in last]):.1%} "
                    f"active={state['elapsed_s']:.1f}s")
                save()
            if number in Config.SNAPSHOT_STEPS:
                atomic_checkpoint(folder/f"weights_step_{number:07d}.pt", dict(model=model.state_dict(), identity=identity, step=number))
            if mode == "capacity" and any(state["capacity_passed"]):
                status = "capacity_passed"
                break
            if number % Config.CHECKPOINT_EVERY == 0 and not should_eval:
                save()
    except KeyboardInterrupt:
        STOP = True
        status = "paused"
        log("强制中断：保留最后完整检查点，下次从该点恢复。")
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        generator.set_state(restored["rng"])
        state = restored["state"]
        last = state["history"][-count:]
    state["elapsed_s"] = max(state["elapsed_s"], base+time.perf_counter()-start)
    if not last or last[0]["step"] != state["step"]:
        last = evaluate(model, x, label, data, cid, indices, state, folder, mode)
    save()
    atomic_checkpoint(folder/"final_weights.pt", dict(model=model.state_dict(), identity=identity, step=state["step"]))
    result = dict(mode=mode, case_id=cid, dataset=dataset, step=state["step"], status=status,
                  capacity_passed=any(state["capacity_passed"]), elapsed_s=state["elapsed_s"], models=count, last=last)
    atomic_json(folder/"summary.json", result)
    plot_run(folder, state, mode)
    return result


def classify(results):
    if len(results) != Config.DATASET_REPEATS or any(r["step"] < Config.MAX_STEPS for r in results):
        return "unfinished"
    rows = [row for r in results for row in r["last"]]
    if np.mean([row["train_accuracy"] == 1 for row in rows]) < .9:
        return "training_unresolved"
    means = [np.mean([v["conflict_score"] for v in r["last"]]) for r in results]
    if min(means) > .1 and np.mean([r["conflict_score"] > .05 for r in rows]) >= .8:
        return "A_preferred"
    if max(means) < -.1 and np.mean([r["conflict_score"] < -.05 for r in rows]) >= .8:
        return "B_preferred"
    return "inconclusive"


def package(out):
    dest = out.with_name(out.name+"_package.zip")
    temporary = dest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name != "RUNNING.lock" and not path.name.endswith(".tmp"):
                z.write(path, path.relative_to(out))
    replace_retry(temporary, dest)
    return dest


def main():
    global STOP, LOG_PATH
    STOP = False
    if Config.STAGE not in ("all", "capacity", "duels"):
        raise ValueError("未知STAGE")
    if not torch.cuda.is_available() or torch.device(Config.DEVICE).type != "cuda":
        raise RuntimeError("请使用CUDA版PyTorch，不回落CPU长训")
    if Config.BATCH_SIZE > Config.TRAIN_COUNT:
        raise ValueError("BATCH_SIZE需不大于TRAIN_COUNT")
    signature = science_signature()
    out = Config.OUTPUT_ROOT/f"results_mod97_frozen_function_duels_{Config.RUN_TAG}"
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text(encoding="utf-8")) != signature:
        raise ValueError("目录中科学配置不同，请更换RUN_TAG")
    out.mkdir(parents=True, exist_ok=True)
    lock = out/"RUNNING.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    LOG_PATH = out/"run.log"
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in handlers:
        signal.signal(s, stop_handler)
    torch.set_num_threads(4)
    device = torch.device(Config.DEVICE)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data = load_functions()
    atomic_json(out/"protocol.json", signature)
    atomic_json(out/"panel.json", PANEL)
    atomic_npz(out/"frozen_functions.npz", **data)
    source = globals().get("__file__")
    if source and Path(source).is_file():
        text = Path(source).read_text(encoding="utf-8")
        atomic_text(out/("source_"+hashlib.sha256(text.encode()).hexdigest()[:12]+".py"), text)
    atomic_json(out/f"invocation_{time.time_ns()}.json", dict(gpu=torch.cuda.get_device_name(device),
        torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version(),
        config={k: v for k, v in vars(Config).items() if k.isupper()}))
    deadline = time.perf_counter()+Config.MAX_TOTAL_SECONDS
    verdicts = []
    def capacity(cid):
        path = out/"capacity"/("target" if cid < 0 else f"case_{cid:02d}")
        return run(path, "capacity", cid, -1, data, device, signature, deadline)["capacity_passed"]
    def duel(cid):
        if not capacity(cid):
            return dict(case_id=cid, verdict="capacity_or_optimization_unresolved")
        if Config.STAGE == "capacity":
            return dict(case_id=cid, verdict="capacity_passed")
        results = []
        for dataset in range(Config.DATASET_REPEATS):
            if STOP or time.perf_counter() >= deadline or (out/"STOP").exists():
                break
            results.append(run(out/"duels"/f"case_{cid:02d}_data_{dataset:02d}", "duel", cid, dataset,
                               data, device, signature, deadline))
        return dict(case_id=cid, verdict=classify(results), results=results)
    try:
        if not capacity(-1):
            log("最终规则尚未通过全映射拟合检查；保留结果，不开始比较。")
            return
        for cid in Config.PRIMARY_CASE_IDS:
            if STOP or time.perf_counter() >= deadline or (out/"STOP").exists():
                break
            result = duel(cid)
            verdicts.append(result)
            log(f"VERDICT case{cid}: {result['verdict']}")
            fallback = PANEL["cases"][cid]["fallback_case_id"]
            if Config.AUTO_FAR_IF_INCONCLUSIVE and result["verdict"] == "inconclusive" and fallback is not None:
                verdicts.append(duel(fallback))
            atomic_json(out/"verdicts.json", verdicts)
    finally:
        try:
            log(f"结果包：{package(out)}")
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)
            lock.unlink(missing_ok=True)
            LOG_PATH = None


if __name__ == "__main__":
    main()

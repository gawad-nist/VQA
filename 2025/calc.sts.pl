"""                                                                                                                                                        
NIST-developed software is provided by NIST as a public service. You may use, copy, and distribute copies of the software in any medium, provided that you 
keep intact this entire notice. You may improve, modify, and create derivative works of the software or any portion of the software, and you may copy and  
distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of  
any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.                           
                                                                                                                                                           
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY OPERATION OF LAW,      
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND DATA ACCURACY. NIST        
NEITHR REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES    
NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,  
RELIABILITY, OR USEFULNESS OF THE SOFTWARE.                                                                                                                
                                                                                                                                                           
You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use,    
including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment,     
and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury  
or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.                           
"""

#! /usr/bin/perl

# Given 2 sentences from the command line, returns a score representing how similar

# this is an API service http://swoogle.umbc.edu/SimService/api.html
# PAPER : Lushan Han, Abhay L. Kashyap, Tim Finin, James Mayfield and Johnathan Weese, UMBC_EBIQUITY-CORE: Semantic Textual Similarity Systems, 
# Proc. 2nd Joint Conf. on Lexical and Computational Semantics, Association for Computational Linguistics, June 2013. 

#Usage: get.semantic.similarity.pl -sen children is playing football -sen kids are playing soccer

my $host="swoogle.umbc.edu";
my $protocol="http";


(undef, $s1, $s2) = split /-sen/," @ARGV";

$score = `wget "$protocol://$host/StsService/GetStsSim?operation=api&phrase1=$s1&phrase2=$s2" --user-agent='NIST - TRECVID' -q -O - 2>&1` ;

print "$score";










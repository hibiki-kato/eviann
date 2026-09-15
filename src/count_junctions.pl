#!/usr/bin/env perl
#


my @cds=();
my %jcounts=();
while($line=<STDIN>){
  chomp($line);
  push(@gff,$line);
  my @F=split(/\t/,$line);
  if($F[2] eq "gene"){
    if(scalar(@cds)>1){
      for($i=1;$i<=$#cds;$i++){
        my ($p1,$p2)=split(/\s/,$cds[$i-1]);
        my ($c1,$c2)=split(/\s/,$cds[$i]);
        $jcounts{"$seq $p2 $c1"}++;
      }
    }
    @cds=();
    @f=split(/;/,$F[8]);
    $seq=$F[0];
  }elsif(uc($F[2]) eq "CDS"){
    push(@cds,"$F[3] $F[4]");
  }
}

foreach $line (@gff){
  my @F=split(/\t/,$line);
  if($F[2] eq "gene"){
    $min_count=1000000000;
    if(scalar(@cds)>1){
      for($i=1;$i<=$#cds;$i++){
        my ($p1,$p2)=split(/\s/,$cds[$i-1]);
        my ($c1,$c2)=split(/\s/,$cds[$i]);
        $min_count=$jcounts{"$seq $p2 $c1"} if($jcounts{"$seq $p2 $c1"}<$min_count);
      } 
    }
    $min_gene_count{$id}=$min_count;
    @cds=();
    @f=split(/;/,$F[8]);
    $id=substr($f[0],3);
    $seq=$F[0];
  }elsif(uc($F[2]) eq "CDS"){
    push(@cds,"$F[3] $F[4]");
  }
}

foreach $line (@gff){
  my @F=split(/\t/,$line);
  if($F[2] eq "gene"){
    $F[2]="transcript";
    @f=split(/;/,$F[8]);
    $id=substr($f[0],3);
    $flag=($min_gene_count{$id}>$ARGV[0]) ? 1 : 0;
  }
  print join("\t",@F),";min_j_count=$min_gene_count{$id}\n" if($flag);
}


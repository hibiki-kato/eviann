#!/usr/bin/env perl

my $min_exons=0;
$min_exons=$ARGV[0] if($ARGV[0]>0);
my $gene="";
my $tid="";
my $gid="";
my %beg5UTR;
my %beg3UTR;
my %end5UTR;
my %end3UTR;
my @threeUTR=();
my @fiveUTR=();
while($line=<STDIN>){
  #print "DEBUG $line";
  chomp($line);
  @gff_fields=split(/\t/,$line);
  if($gff_fields[2] eq "mRNA"){
    @gff_fields_t=split(/\t/,$tline);
    if(scalar(@fiveUTR)>$min_exons){
      $gff_fields_t[3]=$beg5UTR{$tid};
      $gff_fields_t[4]=$end5UTR{$tid};
      print join("\t",@gff_fields_t[0..7]),"\tID=$tid.5p\n";
      print join("\n",@fiveUTR),"\n";
    }
    if(scalar(@threeUTR)>$min_exons){
      $gff_fields_t[3]=$beg3UTR{$tid};
      $gff_fields_t[4]=$end3UTR{$tid};
      print join("\t",@gff_fields_t[0..7]),"\tID=$tid.3p\n";
      print join("\n",@threeUTR),"\n";
    }
    @threeUTR=();
    @fiveUTR=();
    $tid=$1 if($gff_fields[8] =~ /evidence_transcript_id=(\S+);start_codon=/);
    $tline=$line;
    $end5UTR{$tid}=0;
    $end3UTR{$tid}=0;
  }elsif($gff_fields[2] eq "five_prime_UTR"){
    $gff_fields[2]="exon";
    $gff_fields[8]="Parent=$tid.5p";
    push(@fiveUTR,join("\t",@gff_fields));
    $beg5UTR{$tid}=$gff_fields[3] unless(defined($beg5UTR{$tid}));
    $end5UTR{$tid}=$gff_fields[4] if($end5UTR{$tid}<$gff_fields[4]);
  }elsif($gff_fields[2] eq "three_prime_UTR"){
    $gff_fields[2]="exon";
    $gff_fields[8]="Parent=$tid.3p";
    push(@threeUTR,join("\t",@gff_fields));
    $beg3UTR{$tid}=$gff_fields[3] unless(defined($beg3UTR{$tid}));
    $end3UTR{$tid}=$gff_fields[4] if($end3UTR{$tid}<$gff_fields[4]);
  }
}


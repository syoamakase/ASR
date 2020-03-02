#! /usr/bin/perl

#open(f, "/n/rd32/mimura/e2e/data/original/script/aps_sps/bpe/bpe.id");
#open(f, "/n/work1/ueno/data/tedlium2/vocab.id");
open(f, "/n/work1/ueno/data/tedlium2/bpe1k/vocab.id");
#open(f, "/n/work1/ueno/data/tedlium2/bpe1k/old/vocab.id");
#open(f, "/n/work1/ueno/data/tedlium2/bpe500/vocab.id");
#open(f, "/n/rd32/mimura/e2e/data/script/aps/word.id");
#open(f, "/n/rd32/mimura/e2e/data/script/sps/word.id");
#open(f, "/n/rd32/mimura/e2e/data/script/aps_sps/word.id");
#open(f, "/n/rd23/ueno/LM/e2e_lm/data/aps_sps/word_aps_sps2aps.id_withword_euc");
#open(f, "/n/rd32/mimura/e2e/data/script/aps/word.id");
#open(f, "/n/rd32/mimura/e2e/data/script/aps_sps/char-wb.id");
#open(f, "/n/rd32/mimura/e2e/data/script/aps_sps/word.id");

while(<f>){
    chomp;
    @a = split;
    $char{@a[1]} = @a[0];
}
close(f);

print "#!MLF!#\n";
while(<>){
    chomp;
    @a = split;
    $id = @a[0];
    if($id =~ /\/([^\/]+)\.wav/){$utt = $1;}
    if($id =~ /\/([^\/]+)\.htk/){$utt = $1;}
    if($id =~ /\/([^\/]+)\.npy/){$utt = $1;}
    print "\"$utt.htk\"\n";
    shift(@a);
    for(@a){print "$char{$_}\n";}
    print ".\n";
}

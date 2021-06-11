root=$(readlink --canonicalize $(dirname $0)) # absolute path of this script
echo root = $root
for ckpt in $root/xlmr-large..*dist_l1ndotn..hs-0..bn-1*/*/nen-nen-weights; do
	bash predict_scores.sh $root/${ckpt}
done
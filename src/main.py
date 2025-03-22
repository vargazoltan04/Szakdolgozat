import separator.ocr as ocr
import separator.row as row
import separator.character as character

import textdistance
import numpy as np
import argparse
from pathlib import Path


from separator import *
from separator.row_segmentator.row_segmentator_new import RowSegmentatorNew
from util import util

from network.model import VGG16

input1 = """
Lorem ipsum dolor sit amet consectetur adipisicing elit. Numquam ad nulla repellendus maiores, officia alias tenetur, inventore quae quaerat nesciunt iste omnis, amet a esse at reiciendis fugit iure commodi. Veniam voluptatem provident vero ullam iste fugit, laboriosam numquam vel eos, repellendus placeat facilis deleniti, animi laudantium nam perspiciatis aspernatur. Tenetur et consectetur ad nihil nulla sunt quos, esse vel. Dolores amet voluptas adipisci fugiat ut ab rerum repudiandae nemo aspernatur inventore ullam a vitae, ducimus illo qui ea, illum neque? Maiores, doloremque sint vero odio consectetur hic rerum rem! Dicta libero illum quos, quam dolor dignissimos fugiat eum aliquid ipsa quidem aperiam unde eius qui excepturi iste magnam alias perferendis, architecto illo voluptatibus dolores! Nam illo voluptas beatae nobis? Voluptatibus recusandae voluptates ipsum. Voluptatum repellat blanditiis dolor quod, voluptatem dolores quo reiciendis modi labore perspiciatis eligendi dolorum quasi quia? Dolorem adipisci officia omnis eum quos neque ea, recusandae minus! Eos, saepe dicta rerum eveniet doloribus, quibusdam laboriosam consequuntur possimus iste veniam pariatur assumenda, temporibus ex ipsa id delectus? Necessitatibus ducimus obcaecati nihil sed sunt, incidunt vitae eaque blanditiis possimus. Deserunt impedit suscipit repellendus, est quo, dolor optio sequi, mollitia quasi fugit asperiores harum nesciunt ea quidem neque rerum delectus earum exercitationem ipsam. Repellendus consequatur id ipsam sapiente ea ex. Autem reprehenderit in cumque aliquid excepturi, nam magni accusamus molestiae reiciendis eos ex veniam delectus totam consectetur vero mollitia dicta sit. Beatae, deserunt! Saepe quibusdam corrupti aspernatur expedita autem accusantium. Nesciunt eligendi ex mollitia similique ipsum, et itaque sint laboriosam, ut sapiente cum ipsam ullam reprehenderit vero eos culpa facere debitis omnis ad aliquam! Repellat sunt delectus ea! Aspernatur, nostrum. Quasi, est. Voluptates, quia neque illo cupiditate, necessitatibus beatae rem optio adipisci iste porro dolor voluptatem hic tempore obcaecati sapiente quos similique quod mollitia. Nulla sint deserunt pariatur qui assumenda.
"""
input2 = """
Lorem, ipsum dolor sit amet consectetur adipisicing elit. Tempore quos laboriosam sequi dignissimos iure veniam est, quaerat aperiam nesciunt quae placeat animi voluptatibus excepturi et similique, numquam quo praesentium voluptas accusantium culpa doloribus. Ipsum cumque consectetur dolorem molestiae assumenda exercitationem tenetur delectus, esse vitae eum odit? Laudantium at iusto recusandae perspiciatis veniam qui libero id. Dolore deleniti libero iste atque porro exercitationem consectetur minus non dolorem a quaerat enim illum accusamus odit, asperiores ab, nemo numquam, vel velit aspernatur minima quas culpa dolor repellendus? Facere explicabo vero modi quaerat voluptatem cupiditate velit dignissimos, fuga officiis repellendus sed commodi maxime iste quisquam laboriosam voluptatibus sequi a iure nobis illo minus. Ea iusto in inventore sint iure quasi nulla quas error, ut qui incidunt ullam libero dolor accusantium quos perspiciatis amet dolore provident. Sapiente quod non nemo excepturi minima similique impedit cumque laudantium laboriosam ad reiciendis ut, aut voluptas sed autem quasi, facilis aliquid blanditiis fugiat ea eaque cupiditate delectus. Aperiam ducimus assumenda sit vero temporibus placeat animi accusantium eius voluptatum rerum, dolor error neque quisquam aut ipsam nostrum molestiae veritatis, cum ex perspiciatis excepturi. Nam dignissimos magnam sed dicta autem vitae illo vero! Laborum dolore unde in ab tempora porro quidem rem illo fuga sint, laudantium alias repellat, dolorem, illum nulla esse! Quisquam excepturi ullam error similique nihil, asperiores consequatur aspernatur magni, pariatur placeat officia. Voluptas, atque eaque distinctio nisi dolorem, mollitia dicta maxime, praesentium illum qui autem nihil sapiente debitis voluptatum beatae soluta consequatur eum dolorum odit ad adipisci. Laudantium tenetur praesentium quam nostrum voluptatum quos reiciendis blanditiis id, vero eius minus veritatis. Accusantium cum tempore deleniti deserunt a nulla officia accusamus, quod aperiam, provident consequatur nobis. Dolore omnis quis tempora voluptas et ullam sit nihil corporis facilis, nesciunt hic aspernatur non culpa consequuntur possimus amet cum, doloremque harum nulla perspiciatis? At pariatur, inventore animi quidem omnis adipisci rem totam incidunt quo quis sequi ex aperiam, ipsa quas! Fugit, totam maxime repellat velit autem quisquam, dolore qui repudiandae nesciunt, rem assumenda veritatis nostrum? Sint amet ad facilis blanditiis debitis saepe delectus in reprehenderit quidem mollitia? Quis eligendi impedit sed, laborum totam fugiat quia maiores earum vitae iste culpa harum vel reiciendis voluptates! Aperiam magni, quam, veniam voluptas earum numquam, obcaecati totam mollitia saepe est a. Rem voluptas quasi ipsum recusandae id, alias adipisci? Id facere, illum veniam cum modi eaque tenetur facilis doloremque at labore non blanditiis. Rem aperiam voluptatibus doloribus? Rem impedit nesciunt quia vero et aliquam eligendi harum alias facilis porro non voluptates explicabo maiores quod, rerum dolorum? Velit ab explicabo maiores voluptatum quae ad laudantium laborum, qui quis modi ut, veniam quibusdam? Molestiae doloribus mollitia dolores, facere maiores sunt doloremque accusamus voluptatibus iste, rerum suscipit possimus illum aliquid, neque repudiandae vero nam temporibus modi quaerat ut vitae dicta dolorem debitis blanditiis? Tempora facere dolorem assumenda doloribus saepe, quae dicta cumque, velit commodi ab repellendus debitis, quo quod modi! Placeat voluptatibus, harum quae et eligendi, veniam corporis dolore tempora porro dolor accusamus natus quam qui obcaecati molestias nam.
"""
input3 = """
Lorem, ipsum dolor sit amet consectetur adipisicing elit. Recusandae aut quisquam eaque aspernatur, odit iusto nulla consequuntur nihil eveniet harum, quia dicta? Adipisci, amet dicta pariatur natus dolor omnis accusantium reprehenderit minus eius sint id quod quas quos distinctio ipsam rerum ea officiis laborum corporis laudantium aliquid saepe? Consequuntur aspernatur obcaecati ipsam inventore nihil neque, odit sit recusandae culpa similique adipisci iusto in excepturi voluptatum totam ex iste harum sunt a? Sunt, iusto? Temporibus maiores pariatur molestias iste, nemo facere autem dicta eveniet est incidunt optio repellat doloremque recusandae porro nulla, at vitae minima cupiditate hic. Ea labore deleniti non nesciunt, est, eaque facere laborum enim inventore aliquid saepe vel, nobis ipsum cumque. Dignissimos quia dolores vel odit consectetur temporibus nisi, eos quis voluptatibus tenetur omnis, expedita perferendis ducimus similique reprehenderit ratione veniam. Veniam ut quod totam modi qui, incidunt cum ad recusandae quaerat similique accusantium sed reprehenderit, assumenda eaque corrupti repellendus accusamus rem. Accusantium aspernatur mollitia voluptas unde ipsa quo eligendi itaque dolores temporibus, porro voluptatum tempore sapiente ad necessitatibus magnam saepe exercitationem velit sequi numquam quos alias! Repudiandae aut consequuntur eligendi iste labore omnis delectus maiores rem esse ea harum commodi hic tempora fugit itaque beatae aliquid error sed, consequatur, fuga consectetur vel incidunt officiis! Laboriosam quibusdam veniam laborum ipsum, animi deleniti blanditiis sequi qui, voluptatum saepe officiis fugit in similique unde assumenda illo quod voluptatem dolores dolore quia nemo, non id vero. Enim magnam illo repellendus mollitia deleniti? Id consequuntur dicta debitis distinctio! Possimus vero quaerat doloremque doloribus sunt ipsum recusandae libero deleniti sequi, sed aperiam sapiente impedit voluptas voluptatem fugit quia repudiandae fuga officia enim. Architecto nisi reiciendis non id veritatis corrupti! Nesciunt consequatur sint at reprehenderit ratione expedita adipisci harum nostrum explicabo debitis maiores veritatis ut animi, eum laboriosam perferendis earum numquam delectus consectetur sed recusandae, id velit. Qui vero voluptatum ducimus quaerat? Vero id beatae pariatur obcaecati inventore. Explicabo quaerat quidem rerum, odit a, similique, cum dolor culpa quis cumque ad possimus quos totam soluta tenetur placeat! Neque magni dolor unde ex incidunt eligendi inventore iusto. Dignissimos, error deleniti eos quaerat veniam nulla, ut reprehenderit id provident cum, dolor dolorum. Officiis quae atque quaerat nostrum aspernatur, magnam amet veniam eaque, adipisci mollitia perspiciatis expedita accusamus nam reiciendis. Facere ullam consequuntur quos vero delectus neque hic porro sint ratione alias? Minima eum alias laudantium eaque numquam fugiat non dolorum tempore magnam iste? Error tempore, a quidem unde molestiae dicta placeat sequi hic non saepe fugiat voluptate! Quo, nesciunt ratione atque nisi pariatur quae nulla, quod aliquam obcaecati, quibusdam blanditiis incidunt dolores eaque! Alias esse ipsam velit porro eligendi tenetur magnam, est numquam. A commodi aut vel provident iusto nesciunt laborum unde voluptates ad, adipisci id eum nostrum illo sapiente quod recusandae earum est labore iste? Ea non voluptatem in odit at dolor consequatur nulla excepturi temporibus, mollitia rem, est ipsam nobis voluptas animi quasi obcaecati consequuntur debitis, doloremque dignissimos magni assumenda laboriosam adipisci deserunt. Quidem recusandae assumenda illum fugiat excepturi facere quisquam repudiandae possimus facilis!
"""



def main(path, output_path):
    recognizer = Recognizer(58, "./network/model_weights_58_15-35-55_kisnagybetu.pth", "./network/index_class_mapping.json")
    binarizer = BinarizerThresh()
    cleaner = Cleaner()
    row_segmentator = RowSegmentator()
    letter_segmentator = LetterSegmentator()
    resizer = Resizer(45)

    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, path, output_path)
    OCR.run()
    


   
    labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,-: "
    visualizer: BaseVisualizer = Visualizer("../images/output/")
    # cm = visualizer.generate_confusion_matrix(labels, input1, output1, True)
    # cm += visualizer.generate_confusion_matrix(labels, input2, output2, True)
    cm = visualizer.generate_confusion_matrix(labels, input3, OCR.get_output(), True)
    visualizer.plot_confusion_matrix(cm, labels, True, output_path)
    
    print("Levenshtein távolság:", textdistance.levenshtein(input3, OCR.get_output()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help = "The path of the image")
    parser.add_argument("-op", "--output_path", help = "The output path of the result")
    args = parser.parse_args()

    if args.path and args.output_path:
        print("The input image: %s" % args.path)
        print("The output path: %s" % args.output_path)
        util.create_path(args.output_path)
        main(args.path, args.output_path)
    if args.path and not args.output_path:
        print("The input image: %s" % args.path)
        print("The output path: %s" % args.path)
        util.create_path(args.path)

        path_obj = Path(args.path)
        if path_obj.suffix:
            main(args.path, path_obj.parent)
        else:
            print("Give me an input image!")

    

    print("Give me a path that I can recognize.")
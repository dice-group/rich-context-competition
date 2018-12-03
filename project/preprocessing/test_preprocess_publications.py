from project.preprocessing.preprocess_publications import PublicationPreprocessing


obj = PublicationPreprocessing()

def test_gather_nounPhrases():
    text1 = 'Some text is here.'
    expect1 = ['Some text']
    res1 = obj.gather_nounPhrases(text1)
    assert res1 == expect1


def test_extract_paragraphs():
    text1 = 'This is a random\nparagraph about something very interesting. It does not have many sentences.\nHowever, it makes a good test.'
    res1 = obj.extract_paragraphs(text1)
    expect1 = ['This is a random paragraph about something very interesting. It does not have many sentences.', 'However, it makes a good test.']
    assert res1 == expect1


def test_dehyphenation_and_noise_removal():
    text1 = 'This is 123 testin-\ng of sentences.\nxct'
    res1 = obj.remove_noise_and_handle_hyphenation(text1,dehyphenation=True)
    expect1 = 'This is 123 testing\n of sentences.'
    assert res1 == expect1

    text2 = 'Rules to be\n followed: 1. No servings. Only self-\nservice.'
    res2 = obj.remove_noise_and_handle_hyphenation(text2, dehyphenation=True)
    expect2 = 'Rules to be\n followed: 1. No servings. Only self-service.'
    assert res2 == expect2

    text3 = 'xit\n=\nxit\n- xit-1\n0.5 (xit\n+ xit-1)\n, (1)'
    res3 = obj.remove_noise_and_handle_hyphenation(text3, dehyphenation=True)
    expect3 = ''
    assert res3 == expect3

    text4 = 'Heinz Herrmann\nChristoph Memmel\nDeutsche Bundesbank, Wilhelm-Epstein-Straße 14, 60431 Frankfurt am Main,\nPostfach 10 06 02, 60006 Frankfurt am Main\nTel +49 69 9566-0\nTelex within Germany 41227, telex from abroad 414431'
    res4 = obj.remove_noise_and_handle_hyphenation(text4)
    expect4 = 'Heinz Herrmann\nChristoph Memmel\nDeutsche Bundesbank, Wilhelm-Epstein-Straße 14, 60431 Frankfurt am Main,'
    assert res4 == expect4

    text5 = "yikjt,out\n= 1Yout\n+ 2xit"
    res5 = obj.remove_noise_and_handle_hyphenation(text5)
    expect5 = 'yikjt,out'
    assert res5 == expect5

    text6 = "2. A Simple Model of Trade Openness and Output Volatility..................................... 4\n2.1 Exposure and Reaction to Shocks .......................................................................... 4"
    res6 = obj.remove_noise_and_handle_hyphenation(text6)
    expect6 = ''
    assert res6 == expect6


def test_intro_extraction():
    text1 = 'Contents\n1.Introduction\n2.Some other thing\n1. Introduction\nJust to test what is picked.\n'
    res1 = obj.find_introduction(text1)
    expect1 = text1.rfind('1. Introduction')
    assert res1 == expect1


def test_references():
    text = 'just to find references or References.\nLITERATURE CITED\nwork 1, work2'
    res = obj.remove_references(text)
    expect = 'just to find references or References.\n'
    assert res == expect


def find_abstract():
    text = 'The Abstractive Summarization of Poltics.\nabstract'
    res = obj.find_abstract(text)
    expect = text.rfind('abstract')
    assert res == expect

    text = 'abstract:Objective of this work is\nto explicitly potray\nAbstract extraction.'
    res = obj.find_abstract(text)
    expect = text.find('abstract')
    assert res == expect
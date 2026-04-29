"""
bitcoin_kv_store.py
CS469 Big Data Analytics — Unit 3 Individual Project
Author: Franklin Woodard

A key-value store of Bitcoin/blockchain concepts demonstrating dictionary
operations analogous to MapReduce patterns on Hadoop.
"""

# ── Dictionary: Bitcoin concept (key) → definition (value) ───────────────────
bitcoin_glossary = {
    1:  "Hashrate: The total computational power being used to mine and process Bitcoin transactions, measured in exahashes per second (EH/s).",
    2:  "Mining Difficulty: A measure of how hard it is to find a valid block hash, adjusted every 2016 blocks (~2 weeks) to target a 10-minute block time.",
    3:  "Block Height: The number of blocks preceding a given block in the blockchain, starting at 0 (the Genesis Block).",
    4:  "Node: A computer that participates in the Bitcoin network by holding a full copy of the blockchain and independently validating all transactions.",
    5:  "Mempool: The memory pool where unconfirmed transactions wait to be included in a block by miners.",
    6:  "Satoshi: The smallest unit of Bitcoin, equal to 0.00000001 BTC, named after Bitcoin's pseudonymous creator Satoshi Nakamoto.",
    7:  "UTXO: Unspent Transaction Output — the fundamental building block of Bitcoin's accounting model; every transaction consumes UTXOs and creates new ones.",
    8:  "Proof of Work: The consensus mechanism requiring miners to expend computational energy to propose valid blocks, making attacks prohibitively expensive.",
    9:  "Merkle Tree: A binary tree of hashes used to efficiently summarize and verify all transactions in a block.",
    10: "Genesis Block: The first block of the Bitcoin blockchain, mined by Satoshi Nakamoto on January 3, 2009.",
    11: "Halving: The programmatic reduction of the block subsidy by 50% every 210,000 blocks (~4 years), controlling Bitcoin's supply issuance.",
    12: "Lightning Network: A Layer 2 payment protocol built on Bitcoin enabling fast, low-fee transactions through payment channels without on-chain settlement.",
    13: "SegWit: Segregated Witness — a 2017 protocol upgrade that moved signature data outside the main transaction block, increasing capacity and fixing transaction malleability.",
    14: "Taproot: A 2021 upgrade enabling Schnorr signatures and MAST, improving privacy and smart contract efficiency on Bitcoin.",
    15: "Schnorr Signature: A digital signature scheme adopted in Taproot offering smaller signatures, key aggregation, and improved privacy over ECDSA.",
    16: "ECDSA: Elliptic Curve Digital Signature Algorithm — the cryptographic scheme used in Bitcoin for signing transactions prior to Taproot.",
    17: "HD Wallet: Hierarchical Deterministic wallet — derives an unlimited number of key pairs from a single seed phrase using BIP-32.",
    18: "BIP-32: Bitcoin Improvement Proposal defining the HD wallet derivation standard used by virtually all modern Bitcoin wallets.",
    19: "BIP-39: Bitcoin Improvement Proposal defining the 12- or 24-word mnemonic seed phrase standard for backing up HD wallets.",
    20: "Timelock: A transaction constraint preventing Bitcoin from being spent until a specified block height or Unix timestamp is reached.",
    21: "HTLC: Hash Time Locked Contract — a smart contract primitive using hashlocks and timelocks to enable trustless atomic swaps and Lightning channels.",
    22: "Coinbase Transaction: The first transaction in every block, created by the miner to claim the block subsidy and transaction fees.",
    23: "Block Subsidy: The newly issued Bitcoin awarded to miners per block, currently 3.125 BTC following the April 2024 halving.",
    24: "Difficulty Adjustment: The automatic recalibration of mining difficulty every 2016 blocks to maintain the ~10-minute block interval target.",
    25: "Orphan Block: A valid block that is not included in the main chain because a competing block at the same height was accepted by the network first.",
    26: "51% Attack: A scenario where a single entity controls more than half of the network hashrate, theoretically enabling double-spend attacks.",
    27: "Double Spend: An attempt to spend the same Bitcoin twice by broadcasting conflicting transactions; prevented by Proof of Work and block confirmations.",
    28: "Confirmations: The number of blocks added after the block containing a transaction; more confirmations mean greater finality and security.",
    29: "Fee Rate: The transaction fee per unit of block space (sat/vByte), used by miners to prioritize transactions during high-demand periods.",
    30: "Replace-By-Fee (RBF): A protocol allowing unconfirmed transactions to be replaced with a higher-fee version, improving fee estimation flexibility.",
    31: "Child-Pays-For-Parent (CPFP): A fee-bumping technique where a child transaction pays a high enough fee to incentivize miners to include its low-fee parent.",
    32: "Script: Bitcoin's stack-based, non-Turing-complete scripting language that defines the conditions under which UTXOs can be spent.",
    33: "P2PKH: Pay-to-Public-Key-Hash — the original Bitcoin address format (starting with '1'), locking funds to a hashed public key.",
    34: "P2SH: Pay-to-Script-Hash — address format (starting with '3') enabling multisig and complex spending conditions encoded in a redeem script.",
    35: "P2WPKH: Pay-to-Witness-Public-Key-Hash — native SegWit address format (starting with 'bc1q') for single-sig outputs.",
    36: "P2TR: Pay-to-Taproot — the newest address format (starting with 'bc1p') enabled by the Taproot upgrade, supporting key-path and script-path spending.",
    37: "Multisig: A script requiring M-of-N signatures to authorize a transaction, used for shared custody and enhanced security.",
    38: "Cold Storage: Keeping Bitcoin private keys offline (hardware wallet, paper wallet, air-gapped computer) to protect against remote attacks.",
    39: "Hardware Wallet: A dedicated physical device that stores private keys offline and signs transactions without exposing keys to networked computers.",
    40: "Seed Phrase: A human-readable sequence of 12 or 24 words encoding the master private key of an HD wallet; the ultimate backup for all funds.",
    41: "SPV: Simplified Payment Verification — a lightweight client mode that verifies transactions using block headers only, without downloading the full blockchain.",
    42: "Bloom Filter: A probabilistic data structure used by SPV clients to request relevant transactions from full nodes without revealing their addresses.",
    43: "Compact Block: A relay optimization (BIP-152) that sends block headers and short transaction IDs instead of full transactions, reducing propagation latency.",
    44: "Stratum Protocol: The dominant mining pool communication protocol used to distribute work from pool servers to individual miners.",
    45: "Mining Pool: A group of miners who combine hashrate and share block rewards proportionally, smoothing out individual income variance.",
    46: "Nonce: A 32-bit number that miners increment repeatedly when searching for a block hash below the current difficulty target.",
    47: "ExtraNonce: Additional nonce space in the coinbase transaction used when the 32-bit nonce space is exhausted during mining.",
    48: "Target: The numerical value that a valid block hash must be less than or equal to; lower target means higher difficulty.",
    49: "Block Header: The 80-byte metadata structure hashed during mining, containing: version, previous block hash, Merkle root, timestamp, target bits, and nonce.",
    50: "Chain Work: The cumulative Proof of Work across all blocks in a chain, used by nodes to determine which chain represents the most work (longest chain rule).",
    51: "Immutability: The property that once a transaction is buried under sufficient blocks, altering it would require redoing all subsequent Proof of Work.",
    52: "Byzantine Fault Tolerance: The ability of the Bitcoin network to reach consensus even when some participants (nodes or miners) behave maliciously or fail.",
    53: "Gossip Protocol: The peer-to-peer method by which Bitcoin nodes propagate transactions and blocks across the network by relaying to connected peers.",
    54: "Compact Filters (BIP-158): A modern alternative to Bloom Filters allowing lightweight clients to privately download only relevant block data.",
    55: "Timechain: An alternative term for the Bitcoin blockchain emphasizing its role as an immutable, time-stamped ledger secured by Proof of Work.",
}

SEPARATOR = "─" * 65

# ── 1. Enumerate all key-value pairs ─────────────────────────────────────────
def enumerate_glossary(glossary):
    print(f"\n{'ENUMERATE: All Key-Value Pairs':^65}")
    print(SEPARATOR)
    for idx, (key, value) in enumerate(glossary.items(), start=1):
        print(f"[{idx:>2}] Key {key:>2}: {value[:70]}{'...' if len(value) > 70 else ''}")
    print(f"\nTotal entries: {len(glossary)}")

# ── 2. List all keys ──────────────────────────────────────────────────────────
def list_keys(glossary):
    print(f"\n{'ALL KEYS':^65}")
    print(SEPARATOR)
    keys = list(glossary.keys())
    print(f"Keys: {keys}")
    print(f"\nTotal keys: {len(keys)}")

# ── 3. List all values ────────────────────────────────────────────────────────
def list_values(glossary):
    print(f"\n{'ALL VALUES (first 60 chars each)':^65}")
    print(SEPARATOR)
    for key, value in glossary.items():
        print(f"  Key {key:>2} → {value[:60]}...")
    print(f"\nTotal values: {len(glossary)}")

# ── 4. Replace value for key 1 ────────────────────────────────────────────────
def replace_key_one(glossary):
    print(f"\n{'REPLACE: Key 1 Value':^65}")
    print(SEPARATOR)
    old_value = glossary[1]
    new_value = (
        "Hashrate (Updated): The aggregate computational power of the "
        "entire Bitcoin network, currently exceeding 700 EH/s — the "
        "highest in Bitcoin's history — representing the total security "
        "budget enforced by Proof of Work. No government, corporation, "
        "or adversary can rewrite history without matching this energy output."
    )
    print(f"Before → Key 1: {old_value[:70]}...")
    glossary[1] = new_value
    print(f"After  → Key 1: {glossary[1][:70]}...")
    print("\nKey 1 successfully updated.")
    return glossary

# ── 5. MapReduce analogy demo ─────────────────────────────────────────────────
def mapreduce_word_count(glossary):
    """
    Demonstrates a MapReduce-style word frequency count across all values.
    Map phase:  emit (word, 1) for every word in every value
    Reduce phase: sum counts per word
    This mirrors how Hadoop would distribute this across worker nodes.
    """
    print(f"\n{'MAPREDUCE ANALOGY: Word Frequency (Top 15)':^65}")
    print(SEPARATOR)

    # MAP: tokenize each value into (word, 1) pairs
    mapped = []
    for value in glossary.values():
        words = value.lower().replace(",", "").replace(".", "").replace("—", "").split()
        for word in words:
            if len(word) > 4:           # skip noise words
                mapped.append((word, 1))

    # SHUFFLE + REDUCE: aggregate counts per key
    reduced = {}
    for word, count in mapped:
        reduced[word] = reduced.get(word, 0) + count

    # Sort and display top 15
    top_words = sorted(reduced.items(), key=lambda x: x[1], reverse=True)[:15]
    for rank, (word, count) in enumerate(top_words, 1):
        bar = "█" * count
        print(f"  {rank:>2}. {word:<20} {count:>3}  {bar}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(SEPARATOR)
    print(f"{'BITCOIN GLOSSARY KEY-VALUE STORE':^65}")
    print(f"{'CS469 Big Data Analytics — Unit 3 IP':^65}")
    print(f"{'Franklin Woodard':^65}")
    print(SEPARATOR)

    enumerate_glossary(bitcoin_glossary)
    list_keys(bitcoin_glossary)
    list_values(bitcoin_glossary)
    bitcoin_glossary = replace_key_one(bitcoin_glossary)
    mapreduce_word_count(bitcoin_glossary)

    print(f"\n{SEPARATOR}")
    print("Program complete.")
    print(SEPARATOR)

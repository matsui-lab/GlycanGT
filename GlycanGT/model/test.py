import torch
import torch.nn as nn

import sys
sys.path.append("/share3/kitani/glycoGT/model")
from glycan_tokengt import GlycanTokenGT
from config_tokengt import get_config

def create_dummy_batch(batch_size: int, max_nodes: int, cfg: dict):
    """
    モデルのforwardパステストのためのダミーデータ生成関数
    """
    # 1. 各グラフのノード数とエッジ数をランダムに決定
    node_num = torch.randint(low=5, high=max_nodes + 1, size=(batch_size,))
    edge_num = node_num - 1 

    total_nodes = node_num.sum().item()
    total_edges = edge_num.sum().item()

    # 2. 各キーに対応するダミーテンソルを生成
    node_data = torch.randint(low=1, high=cfg["num_atoms"], size=(total_nodes,))
    edge_data = torch.randint(low=1, high=cfg["num_edges"], size=(total_edges,))

    edge_index_list = []
    for i in range(batch_size):
        n = node_num[i].item()
        e = edge_num[i].item()
        
        dst_nodes = torch.randperm(n - 1) + 1
        dst = dst_nodes[:e]
        
        src = torch.LongTensor([torch.randint(0, d, (1,)).item() for d in dst])
        edge_index_list.append(torch.stack([src, dst]))
    
    edge_index = torch.cat(edge_index_list, dim=1)

    in_degree = torch.zeros(total_nodes, dtype=torch.long)
    out_degree = torch.zeros(total_nodes, dtype=torch.long)
    
    lap_eigvec = torch.randn(total_nodes, 16)
    lap_eigval = torch.randn(total_nodes, 16)

    return {
        "node_data": node_data,
        "edge_data": edge_data,
        "edge_index": edge_index,
        "node_num": node_num.tolist(),
        "edge_num": edge_num.tolist(),
        "in_degree": in_degree,
        "out_degree": out_degree,
        "lap_eigvec": lap_eigvec,
        "lap_eigval": lap_eigval,
    }

def run_test():
    """テストを実行"""
    print("--- TokenGTモデル 統合テスト開始 ---")

    try:
        config = get_config('small')
        print("✅ ステップ1: 'small'モデルの設定の読み込み完了")
        print(f"   (埋め込み次元数: {config['embedding_dim']}, レイヤー数: {config['num_encoder_layers']})")

        model = GlycanTokenGT(cfg=config)
        model.eval()
        print("✅ ステップ2: GlycanTokenGTモデルのインスタンス化に成功")

        dummy_batch = create_dummy_batch(batch_size=4, max_nodes=20, cfg=config)
        print("✅ ステップ3: テスト用のダミーデータの生成に成功")
        print(f"   (バッチサイズ: 4, 総ノード数: {sum(dummy_batch['node_num'])})")
        
        print("🚀 ステップ4: モデルのフォワードパスを実行中...")
        with torch.no_grad():
            logits, extra = model(batch=dummy_batch, task="smtp")
        print("✅ ステップ4: フォワードパスが完了")

        print("🔍 ステップ5: 出力テンソルの形状を確認...")
        expected_batch_size = len(dummy_batch["node_num"])
        expected_vocab_size = config["num_atoms"]
        expected_embed_dim = config["embedding_dim"]

        assert logits.shape[1] == expected_batch_size
        assert logits.shape[2] == expected_vocab_size
        print(f"   - logitsの形状は期待通り: {list(logits.shape)}")

        assert extra['graph_rep'].shape[0] == expected_batch_size
        assert extra['graph_rep'].shape[1] == expected_embed_dim
        print(f"   - graph_repの形状は期待通りです: {list(extra['graph_rep'].shape)}")
        
        print("✅ ステップ5: 出力形状の確認に成功")

    except Exception as e:
        # エラーが発生した場合は、ここでまとめてキャッチして表示
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc() # 詳細なトレースバックを表示
        return

    print("\n🎉🎉🎉 テストが正常に完了 🎉🎉🎉")
    print("作成したモジュール群は、少なくともエラーなく連携して動作する")


if __name__ == "__main__":
    run_test()